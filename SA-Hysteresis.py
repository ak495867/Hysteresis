import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path('/content/sa_hysteresis_v21') if Path('/content').exists() else Path.cwd() / 'sa_hysteresis_v21'
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'figures'
for path in [DATA_DIR, OUT_DIR, FIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

ASSET_GROUPS = {
    'equity_us': ['AAPL','MSFT','AMZN','GOOGL','NVDA','META','TSLA','JPM','V','UNH'],
    'sector': ['XLF','XLK','XLE','XLV','XLI','XLP','XLY','XLC','XLU','XLB'],
    'global_equity': ['EFA','EEM','EWJ','EWG','EWU','EWZ','INDA','FXI','EWY','EWA'],
    'rates_credit': ['TLT','IEF','SHY','LQD','HYG','TIP','AGG','MUB'],
    'commodities_fx': ['GLD','SLV','USO','DBA','UUP','FXE','FXY','FXB'],
    'crypto': ['BTC-USD','ETH-USD','BNB-USD','SOL-USD']
}
ASSETS = [asset for group in ASSET_GROUPS.values() for asset in group]
START = '2016-01-01'
END = None
LOOKBACK = 32
HORIZON = 1
TRAIN_FRAC = 0.60
VALID_FRAC = 0.20
PURGE = LOOKBACK + HORIZON
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
N_FEATURES = 10
LATENT = 24
MEMORY = 16
COST_BPS = 5.0
DEVICE = 'cpu' if os.environ.get('SA_FORCE_CPU', '0') == '1' else ('cuda' if torch.cuda.is_available() else 'cpu')


def download_market_data():
    close_path = DATA_DIR / 'close.parquet'
    volume_path = DATA_DIR / 'volume.parquet'
    if close_path.exists() and volume_path.exists():
        return pd.read_parquet(close_path), pd.read_parquet(volume_path)
    raw = yf.download(ASSETS, start=START, end=END, auto_adjust=True, progress=False, group_by='column', threads=True)
    if not isinstance(raw.columns, pd.MultiIndex):
        close = raw[['Close']].rename(columns={'Close': ASSETS[0]})
        volume = raw[['Volume']].rename(columns={'Volume': ASSETS[0]})
    elif 'Close' in raw.columns.get_level_values(0):
        close = raw['Close'].copy()
        volume = raw['Volume'].copy()
    else:
        close = raw.xs('Close', axis=1, level=1).copy()
        volume = raw.xs('Volume', axis=1, level=1).copy()
    close = close.reindex(columns=ASSETS).ffill(limit=5)
    volume = volume.reindex(columns=ASSETS).replace(0, np.nan).ffill(limit=5)
    close.to_parquet(close_path)
    volume.to_parquet(volume_path)
    return close, volume


def build_observations():
    close, volume = download_market_data()
    log_price = np.log(close)
    returns = log_price.diff()
    volatility = returns.rolling(20, min_periods=10).std()
    volume_change = np.log(volume).diff().replace([np.inf, -np.inf], np.nan).rolling(10, min_periods=5).mean()
    drawdown = close / close.rolling(60, min_periods=20).max() - 1.0
    momentum = log_price.diff(5)
    cross_dispersion = returns.std(axis=1).rolling(10, min_periods=5).mean()
    factor_frames = {}
    for group, assets in ASSET_GROUPS.items():
        factor_frames[group] = returns[assets].mean(axis=1)
    factors = pd.DataFrame(factor_frames, index=returns.index)
    market_factor = returns.mean(axis=1).rename('market')
    factors['market'] = market_factor
    factors = factors[['market', 'equity_us', 'rates_credit', 'commodities_fx', 'crypto']]
    factors['dispersion'] = cross_dispersion
    blocks = [returns, volatility, volume_change, drawdown, momentum]
    asset_features = np.stack([block.reindex(columns=ASSETS).values for block in blocks], axis=-1)
    factor_values = factors[['market', 'equity_us', 'rates_credit', 'commodities_fx', 'crypto']].values
    factor_features = np.repeat(factor_values[:, None, :], len(ASSETS), axis=1)
    combined = np.concatenate([asset_features, factor_features], axis=-1)
    data = pd.DataFrame({'date': returns.index, 'dispersion': cross_dispersion.values})
    valid = np.isfinite(combined).all(axis=(1, 2)) & np.isfinite(returns.values).all(axis=1) & np.isfinite(factors.values[:, :5]).all(axis=1)
    combined = combined[valid].astype(np.float32)
    returns_array = returns.values[valid].astype(np.float32)
    factor_array = factors[['market', 'equity_us', 'rates_credit', 'commodities_fx', 'crypto']].values[valid].astype(np.float32)
    dates = returns.index[valid]
    dispersion = factors['dispersion'].values[valid].astype(np.float32)
    np.save(DATA_DIR / 'combined_features.npy', combined)
    np.save(DATA_DIR / 'raw_returns.npy', returns_array)
    np.save(DATA_DIR / 'factor_returns.npy', factor_array)
    np.save(DATA_DIR / 'dispersion.npy', dispersion)
    pd.Series(dates).to_pickle(DATA_DIR / 'dates.pkl')
    return combined, returns_array, factor_array, dispersion, pd.DatetimeIndex(dates)


def make_factor_residuals(returns, factors, train_end):
    n_assets = returns.shape[1]
    n_factors = factors.shape[1]
    design = np.concatenate([np.ones((train_end, 1), dtype=np.float32), factors[:train_end]], axis=1)
    betas = np.zeros((n_assets, n_factors + 1), dtype=np.float32)
    for asset in range(n_assets):
        betas[asset] = np.linalg.lstsq(design, returns[:train_end, asset], rcond=None)[0]
    all_design = np.concatenate([np.ones((len(factors), 1), dtype=np.float32), factors], axis=1)
    fitted = all_design @ betas.T
    residuals = returns - fitted
    return residuals.astype(np.float32), betas.astype(np.float32)


def make_samples(features, returns, factors, dispersion, dates, train_end_raw):
    residuals, betas = make_factor_residuals(returns, factors, train_end_raw)
    targets = np.roll(residuals, -HORIZON, axis=0)
    samples = []
    sample_targets = []
    sample_shocks = []
    sample_dispersion = []
    sample_dates = []
    for i in range(LOOKBACK - 1, len(features) - HORIZON):
        samples.append(features[i - LOOKBACK + 1:i + 1])
        sample_targets.append(targets[i])
        sample_shocks.append(features[i - LOOKBACK + 1:i + 1, :, 0])
        sample_dispersion.append(dispersion[i])
        sample_dates.append(dates[i])
    return np.asarray(samples, dtype=np.float32), np.asarray(sample_targets, dtype=np.float32), np.asarray(sample_shocks, dtype=np.float32), np.asarray(sample_dispersion, dtype=np.float32), pd.DatetimeIndex(sample_dates), betas


def split_arrays(x, y, shocks, dispersion, dates):
    n = len(x)
    train_end = int(n * TRAIN_FRAC)
    valid_end = int(n * (TRAIN_FRAC + VALID_FRAC))
    return {
        'train': (x[:train_end], y[:train_end], shocks[:train_end], dispersion[:train_end], dates[:train_end]),
        'valid': (x[train_end + PURGE:valid_end], y[train_end + PURGE:valid_end], shocks[train_end + PURGE:valid_end], dispersion[train_end + PURGE:valid_end], dates[train_end + PURGE:valid_end]),
        'test': (x[valid_end + PURGE:], y[valid_end + PURGE:], shocks[valid_end + PURGE:], dispersion[valid_end + PURGE:], dates[valid_end + PURGE:])
    }


def scale_splits(splits):
    x_train = splits['train'][0]
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    scale = x_train.reshape(-1, x_train.shape[-1]).std(axis=0) + 1e-8
    result = {}
    for name, values in splits.items():
        x, y, shocks, dispersion, dates = values
        xs = ((x - mean) / scale).astype(np.float32)
        us = xs[:, :, :, 0].astype(np.float32)
        result[name] = (xs, y.astype(np.float32), us, dispersion.astype(np.float32), dates)
    with open(OUT_DIR / 'feature_scaler.json', 'w') as file:
        json.dump({'mean': mean.tolist(), 'scale': scale.tolist()}, file, indent=2)
    return result, mean, scale


class SAHysteresisV21(nn.Module):
    def __init__(self, n_assets, n_features=N_FEATURES, latent=LATENT, memory=MEMORY):
        super().__init__()
        self.n_assets = n_assets
        self.latent = latent
        self.memory = memory
        self.encoder = nn.Sequential(nn.Linear(n_features, 48), nn.GELU(), nn.Linear(48, latent))
        self.alpha_impact = nn.Sequential(nn.Linear(latent + memory * 2, 32), nn.GELU(), nn.Linear(32, memory), nn.Softplus())
        self.beta_impact = nn.Sequential(nn.Linear(latent + memory * 2, 32), nn.GELU(), nn.Linear(32, memory), nn.Sigmoid())
        self.alpha_recovery = nn.Sequential(nn.Linear(latent + memory * 2, 32), nn.GELU(), nn.Linear(32, memory), nn.Softplus())
        self.beta_recovery = nn.Sequential(nn.Linear(latent + memory * 2, 32), nn.GELU(), nn.Linear(32, memory), nn.Sigmoid())
        self.gate = nn.Sequential(nn.Linear(latent + memory * 2, 32), nn.GELU(), nn.Linear(32, 1))
        self.drift = nn.Sequential(nn.Linear(latent + memory * 2, 48), nn.GELU(), nn.Linear(48, latent), nn.Tanh())
        self.impact_q = nn.Linear(latent + memory, 16, bias=False)
        self.impact_k = nn.Linear(latent + memory, 16, bias=False)
        self.recovery_q = nn.Linear(latent + memory, 16, bias=False)
        self.recovery_k = nn.Linear(latent + memory, 16, bias=False)
        self.noise = nn.Sequential(nn.Linear(latent + memory * 2, 32), nn.GELU(), nn.Linear(32, latent), nn.Softplus())
        self.head = nn.Sequential(nn.Linear(latent + memory * 2, 48), nn.GELU(), nn.Linear(48, 4))

    def forward(self, x, shocks):
        batch, steps, assets, features = x.shape
        encoded = self.encoder(x)
        impact_memory = torch.zeros(batch, assets, self.memory, device=x.device)
        recovery_memory = torch.zeros(batch, assets, self.memory, device=x.device)
        gate_last = None
        impact_graph = None
        recovery_graph = None
        state = encoded[:, 0]
        for step in range(steps):
            z = encoded[:, step]
            combined = torch.cat([z, impact_memory, recovery_memory], dim=-1)
            combined = torch.nan_to_num(combined, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            gate = torch.sigmoid(torch.clamp(self.gate(combined), -20.0, 20.0)).squeeze(-1)
            impact_alpha = torch.nan_to_num(self.alpha_impact(combined), nan=0.0, posinf=5.0, neginf=0.0).clamp(0.0, 5.0)
            impact_beta = 0.03 + 0.97 * self.beta_impact(combined).clamp(0.0, 1.0)
            recovery_alpha = torch.nan_to_num(self.alpha_recovery(combined), nan=0.0, posinf=5.0, neginf=0.0).clamp(0.0, 5.0)
            recovery_beta = 0.03 + 0.97 * self.beta_recovery(combined).clamp(0.0, 1.0)
            shock_abs = shocks[:, step].abs().unsqueeze(-1)
            impact_memory = torch.nan_to_num(impact_memory + impact_alpha * shock_abs - impact_beta * impact_memory, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            recovery_memory = torch.nan_to_num(recovery_memory + recovery_alpha * impact_memory - recovery_beta * recovery_memory, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            impact_state = torch.nan_to_num(torch.cat([z, impact_memory], dim=-1), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            recovery_state = torch.nan_to_num(torch.cat([z, recovery_memory], dim=-1), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            impact_q = torch.nan_to_num(self.impact_q(impact_state), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            impact_k = torch.nan_to_num(self.impact_k(impact_state), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            recovery_q = torch.nan_to_num(self.recovery_q(recovery_state), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            recovery_k = torch.nan_to_num(self.recovery_k(recovery_state), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            impact_logits = torch.nan_to_num(torch.einsum('bif,bjf->bij', impact_q, impact_k) / math.sqrt(impact_q.shape[-1]), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            recovery_logits = torch.nan_to_num(torch.einsum('bif,bjf->bij', recovery_q, recovery_k) / math.sqrt(recovery_q.shape[-1]), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            impact_graph = torch.softmax(impact_logits, dim=-1)
            recovery_graph = torch.softmax(recovery_logits, dim=-1)
            impact_flow = torch.einsum('bij,bj->bi', impact_graph, shocks[:, step])
            recovery_flow = torch.einsum('bij,bj->bi', recovery_graph, recovery_memory.norm(dim=-1))
            drift = self.drift(combined)
            scale = self.noise(combined)
            forcing = gate.unsqueeze(-1) * impact_flow.unsqueeze(-1) + (1.0 - gate).unsqueeze(-1) * recovery_flow.unsqueeze(-1)
            state = torch.nan_to_num(z + 0.05 * (drift + scale * forcing), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            gate_last = gate
        final_state = torch.cat([state, impact_memory, recovery_memory], dim=-1)
        output = torch.nan_to_num(self.head(final_state), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        gate_last = torch.nan_to_num(gate_last, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
        return output, gate_last, impact_graph, recovery_graph, impact_memory, recovery_memory


def huber_nll(target, output):
    mean = output[:, :, 0]
    log_scale = torch.clamp(output[:, :, 1], -5.0, 2.0)
    scale = torch.exp(log_scale)
    error = (target - mean) / scale
    nll = 0.5 * error.pow(2) + log_scale
    return nll.mean()


def train_model(train, valid, n_assets):
    x_train, y_train, shock_train, dispersion_train, _ = train
    x_valid, y_valid, shock_valid, dispersion_valid, _ = valid
    model = SAHysteresisV21(n_assets).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_set = TensorDataset(torch.tensor(x_train), torch.tensor(shock_train), torch.tensor(y_train), torch.tensor(dispersion_train))
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    train_dispersion = np.asarray(dispersion_train)
    stress_threshold = float(np.quantile(train_dispersion, 0.75))
    history = []
    best_loss = float('inf')
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb, ub, yb, db in loader:
            xb, ub, yb, db = xb.to(DEVICE), ub.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            output, gate, impact_graph, recovery_graph, impact_memory, recovery_memory = model(xb, ub)
            stress_target = (db >= stress_threshold).float().unsqueeze(-1).expand_as(gate).clamp(0.0, 1.0)
            forecast_loss = huber_nll(yb, output)
            gate_loss = nn.functional.binary_cross_entropy(gate.clamp(1e-6, 1.0 - 1e-6), stress_target)
            impact_sparsity = impact_graph.abs().mean()
            recovery_sparsity = recovery_graph.abs().mean()
            memory_penalty = impact_memory.pow(2).mean() + recovery_memory.pow(2).mean()
            confidence = torch.sigmoid(output[:, :, 3])
            confidence_penalty = ((confidence - torch.exp(-torch.abs(output[:, :, 1]))).pow(2)).mean()
            loss = forecast_loss + 0.05 * gate_loss + 0.005 * impact_sparsity + 0.005 * recovery_sparsity + 0.0005 * memory_penalty + 0.01 * confidence_penalty
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x_valid).to(DEVICE)
            uv = torch.tensor(shock_valid).to(DEVICE)
            yv = torch.tensor(y_valid).to(DEVICE)
            dv = torch.tensor(dispersion_valid).to(DEVICE)
            ov, gv, igv, rgv, imv, rmv = model(xv, uv)
            validation_loss = float(huber_nll(yv, ov).cpu())
            validation_gate = float(nn.functional.binary_cross_entropy(gv.clamp(1e-6, 1.0 - 1e-6), (dv >= stress_threshold).float().unsqueeze(-1).expand_as(gv).clamp(0.0, 1.0)).cpu())
            total_validation = validation_loss + 0.05 * validation_gate
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'valid_loss': total_validation})
        if total_validation < best_loss:
            best_loss = total_validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT_DIR / 'training_history.csv', index=False)
    with open(OUT_DIR / 'training_config.json', 'w') as file:
        json.dump({'stress_threshold': stress_threshold, 'device': DEVICE, 'epochs': EPOCHS, 'lookback': LOOKBACK, 'horizon': HORIZON}, file, indent=2)
    return model


def predict(model, split):
    x, y, shocks, dispersion, dates = split
    model.eval()
    outputs = []
    gates = []
    impact_graphs = []
    recovery_graphs = []
    impact_memories = []
    recovery_memories = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            xb = torch.tensor(x[start:start + BATCH_SIZE]).to(DEVICE)
            ub = torch.tensor(shocks[start:start + BATCH_SIZE]).to(DEVICE)
            output, gate, impact_graph, recovery_graph, impact_memory, recovery_memory = model(xb, ub)
            outputs.append(output.cpu().numpy())
            gates.append(gate.cpu().numpy())
            impact_graphs.append(impact_graph.cpu().numpy())
            recovery_graphs.append(recovery_graph.cpu().numpy())
            impact_memories.append(impact_memory.cpu().numpy())
            recovery_memories.append(recovery_memory.cpu().numpy())
    return {"output": np.concatenate(outputs), "gate": np.concatenate(gates), "impact_graph": np.concatenate(impact_graphs), "recovery_graph": np.concatenate(recovery_graphs), "impact_memory": np.concatenate(impact_memories), "recovery_memory": np.concatenate(recovery_memories)}


def regime_labels(dispersion_train, dispersion_test):
    stress_threshold = float(np.quantile(dispersion_train, 0.75))
    shock_threshold = float(np.quantile(dispersion_train, 0.90))
    labels = np.full(len(dispersion_test), 'calm', dtype=object)
    labels[dispersion_test >= stress_threshold] = 'stress'
    labels[dispersion_test >= shock_threshold] = 'shock'
    for i in range(1, len(labels)):
        if labels[i - 1] in ['shock', 'stress'] and labels[i] == 'calm':
            labels[i] = 'recovery'
    return labels, {'stress_threshold': stress_threshold, 'shock_threshold': shock_threshold}


def positions(prediction, confidence, volatility, threshold=0.25):
    signal = prediction / (volatility + 1e-5)
    confidence = np.clip(confidence, 0.0, 1.0)
    active = confidence >= threshold
    signal = signal * confidence * active
    signal = np.clip(signal, -3.0, 3.0)
    denom = np.sum(np.abs(signal), axis=1, keepdims=True) + 1e-8
    return signal / denom


def portfolio_statistics(prediction, confidence, target, regime_mask=None):
    if regime_mask is None:
        regime_mask = np.ones(len(target), dtype=bool)
    p = prediction[regime_mask]
    c = confidence[regime_mask]
    y = target[regime_mask]
    if len(y) < 2:
        return {'dates': int(len(y)), 'net_sharpe': np.nan, 'net_mean_return': np.nan, 'net_cumulative_return': np.nan, 'turnover': np.nan}
    volatility = np.maximum(np.median(np.abs(p), axis=1, keepdims=True), 1e-5)
    weight = positions(p, c, volatility)
    gross = np.sum(weight * y, axis=1)
    turnover = np.sum(np.abs(np.diff(weight, axis=0)), axis=1)
    costs = np.concatenate([[0.0], turnover * COST_BPS / 10000.0])
    net = gross - costs
    if np.std(net, ddof=1) > 0:
        sharpe = np.sqrt(252.0) * np.mean(net) / np.std(net, ddof=1)
    else:
        sharpe = np.nan
    curve = np.cumprod(1.0 + np.clip(net, -0.25, 0.25))
    return {'dates': int(len(y)), 'net_sharpe': float(sharpe), 'net_mean_return': float(np.mean(net)), 'net_cumulative_return': float(curve[-1] - 1.0), 'turnover': float(np.mean(turnover)) if len(turnover) else 0.0}


def model_metrics(target, prediction, confidence, dates, labels, model_name):
    rows = []
    for regime in ['calm', 'shock', 'stress', 'recovery', 'all']:
        mask = np.ones(len(labels), dtype=bool) if regime == 'all' else labels == regime
        y = target[mask]
        p = prediction[mask]
        c = confidence[mask]
        if len(y) == 0:
            rows.append({'model': model_name, 'regime': regime, 'dates': 0, 'rank_ic': np.nan, 'directional_accuracy': np.nan, 'rmse': np.nan, 'confidence_mean': np.nan, 'net_sharpe': np.nan, 'net_mean_return': np.nan, 'net_cumulative_return': np.nan, 'turnover': np.nan})
            continue
        ics = []
        for t in range(len(y)):
            valid = np.isfinite(y[t]) & np.isfinite(p[t])
            if valid.sum() >= 3:
                ics.append(spearmanr(p[t, valid], y[t, valid]).statistic)
        portfolio = portfolio_statistics(p, c, y)
        rows.append({'model': model_name, 'regime': regime, 'dates': int(len(y)), 'rank_ic': float(np.nanmean(ics)) if ics else np.nan, 'directional_accuracy': float(np.mean(np.sign(p) == np.sign(y))), 'rmse': float(np.sqrt(np.mean((p - y) ** 2))), 'confidence_mean': float(np.mean(c)), **portfolio})
    return rows


def baseline_predictions(train, test):
    x_train, y_train, shocks_train, dispersion_train, dates_train = train
    x_test, y_test, shocks_test, dispersion_test, dates_test = test
    features_train = x_train[:, -1].reshape(len(x_train), -1)
    features_test = x_test[:, -1].reshape(len(x_test), -1)
    ridge = Ridge(alpha=10.0)
    ridge.fit(features_train, y_train)
    forest = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    forest.fit(features_train, y_train)
    history = np.concatenate([y_train, y_test], axis=0)
    test_start = len(y_train)
    persistence = history[test_start - 1:test_start - 1 + len(y_test)]
    momentum = np.stack([history[test_start + i - 5:test_start + i].mean(axis=0) for i in range(len(y_test))], axis=0)
    return {'ridge': ridge.predict(features_test), 'random_forest': forest.predict(features_test), 'persistence': persistence, 'momentum': momentum}


def save_plots(metrics, prediction, confidence, target, labels, dates, output):
    frame = pd.DataFrame(metrics)
    for metric in ['rank_ic', 'directional_accuracy', 'rmse', 'net_sharpe']:
        plot_frame = frame[frame['regime'] != 'all']
        plt.figure(figsize=(11, 6))
        sns.barplot(data=plot_frame, x='regime', y=metric, hue='model')
        plt.axhline(0.0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f'{metric}_by_regime.png', dpi=180)
        plt.close()
    pivot = frame[frame['regime'] != 'all'].pivot(index='model', columns='regime', values='rank_ic')
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', center=0.0, cmap='coolwarm')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rank_ic_heatmap.png', dpi=180)
    plt.close()
    output_prediction = output['output'][:, :, 0]
    gate = output['gate']
    memory = np.linalg.norm(output['impact_memory'], axis=-1).mean(axis=1) + np.linalg.norm(output['recovery_memory'], axis=-1).mean(axis=1)
    plt.figure(figsize=(12, 5))
    plt.plot(dates, gate.mean(axis=1), label='stress_gate')
    plt.plot(dates, memory / (np.max(memory) + 1e-8), label='normalized_memory')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'gate_memory_dynamics.png', dpi=180)
    plt.close()


def main():
    features, raw_returns, factors, dispersion, dates = build_observations()
    raw_train_end = int(len(features) * TRAIN_FRAC)
    x, y, shocks, sample_dispersion, sample_dates, betas = make_samples(features, raw_returns, factors, dispersion, dates, raw_train_end)
    splits = split_arrays(x, y, shocks, sample_dispersion, sample_dates)
    scaled_splits, feature_mean, feature_scale = scale_splits(splits)
    model = train_model(scaled_splits['train'], scaled_splits['valid'], len(ASSETS))
    prediction_output = predict(model, scaled_splits['test'])
    target = scaled_splits['test'][1]
    dates_test = scaled_splits['test'][4]
    dispersion_train = scaled_splits['train'][3]
    dispersion_test = scaled_splits['test'][3]
    labels, thresholds = regime_labels(dispersion_train, dispersion_test)
    prediction = prediction_output['output'][:, :, 0]
    confidence = torch.sigmoid(torch.tensor(prediction_output['output'][:, :, 3])).numpy()
    metrics = model_metrics(target, prediction, confidence, dates_test, labels, 'sa_hysteresis_v21')
    baselines = baseline_predictions(scaled_splits['train'], scaled_splits['test'])
    for name, baseline in baselines.items():
        baseline_confidence = np.ones_like(baseline)
        metrics.extend(model_metrics(target, baseline, baseline_confidence, dates_test, labels, name))
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(OUT_DIR / 'metrics_by_regime.csv', index=False)
    pd.DataFrame({'date': dates_test, 'regime': labels, 'gate': prediction_output['gate'].mean(axis=1), 'impact_memory': np.linalg.norm(prediction_output['impact_memory'], axis=-1).mean(axis=1), 'recovery_memory': np.linalg.norm(prediction_output['recovery_memory'], axis=-1).mean(axis=1)}).to_csv(OUT_DIR / 'state_diagnostics.csv', index=False)
    pd.DataFrame({'date': dates_test, 'regime': labels}).to_csv(OUT_DIR / 'test_regimes.csv', index=False)
    with open(OUT_DIR / 'regime_thresholds.json', 'w') as file:
        json.dump(thresholds, file, indent=2)
    with open(OUT_DIR / 'asset_betas.json', 'w') as file:
        json.dump({asset: betas[i].tolist() for i, asset in enumerate(ASSETS)}, file, indent=2)
    save_plots(metrics, prediction, confidence, target, labels, dates_test, prediction_output)
    print(metrics_frame.to_string(index=False))
    print(json.dumps({'assets': ASSETS, 'n_assets': len(ASSETS), 'device': DEVICE, 'test_dates': len(dates_test), 'regime_counts': pd.Series(labels).value_counts().to_dict(), 'thresholds': thresholds}, indent=2))


if __name__ == '__main__':
    main()
