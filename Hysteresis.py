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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path('/content/hysteresis_research') if Path('/content').exists() else Path.cwd() / 'hysteresis_research'
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'figures'
for p in [DATA_DIR, OUT_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

TICKERS = {
    'equity_us_mega': ['AAPL','MSFT','AMZN','GOOGL','NVDA','META','TSLA','JPM','V','UNH'],
    'equity_sector': ['XLF','XLK','XLE','XLV','XLI','XLP','XLY','XLC','XLU','XLB'],
    'equity_global': ['EFA','EEM','EWJ','EWG','EWU','EWZ','INDA','FXI','EWY','EWA'],
    'rates_credit': ['TLT','IEF','SHY','LQD','HYG','TIP','AGG','MUB'],
    'commodities_fx': ['GLD','SLV','USO','DBA','UUP','FXE','FXY','FXB'],
    'crypto': ['BTC-USD','ETH-USD','BNB-USD','SOL-USD']
}
ASSETS = [x for v in TICKERS.values() for x in v]
START = '2016-01-01'
END = None
LOOKBACK = 32
HORIZON = 1
TRAIN_FRAC = 0.60
VALID_FRAC = 0.20
PURGE = LOOKBACK + HORIZON
BATCH = 128
EPOCHS = 35
LR = 2e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def download_data():
    path = DATA_DIR / 'raw_prices.parquet'
    if path.exists():
        return pd.read_parquet(path)
    data = yf.download(ASSETS, start=START, end=END, auto_adjust=True, progress=False, group_by='column', threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            close = data['Close'].copy()
        else:
            close = data.xs('Close', axis=1, level=1).copy()
        if 'Volume' in data.columns.get_level_values(0):
            volume = data['Volume'].copy()
        else:
            volume = data.xs('Volume', axis=1, level=1).copy()
    else:
        close = data[['Close']].rename(columns={'Close': ASSETS[0]})
        volume = data[['Volume']].rename(columns={'Volume': ASSETS[0]})
    close = close.reindex(columns=ASSETS)
    volume = volume.reindex(columns=ASSETS)
    close = close.ffill(limit=5)
    volume = volume.replace(0, np.nan).ffill(limit=5)
    close.to_parquet(DATA_DIR / 'close.parquet')
    volume.to_parquet(DATA_DIR / 'volume.parquet')
    return pd.read_parquet(DATA_DIR / 'close.parquet')


def load_or_build_features():
    close_path = DATA_DIR / 'close.parquet'
    volume_path = DATA_DIR / 'volume.parquet'
    if not close_path.exists() or not volume_path.exists():
        download_data()
    close = pd.read_parquet(close_path).reindex(columns=ASSETS)
    volume = pd.read_parquet(volume_path).reindex(columns=ASSETS)
    close = close.ffill(limit=5)
    volume = volume.replace(0, np.nan).ffill(limit=5)
    logp = np.log(close)
    ret = logp.diff()
    vol = ret.rolling(20, min_periods=10).std()
    vol_change = np.log(volume).diff().replace([np.inf, -np.inf], np.nan)
    vol_change = vol_change.rolling(10, min_periods=5).mean()
    drawdown = close / close.rolling(60, min_periods=20).max() - 1.0
    momentum = logp.diff(5)
    features = {}
    for name, frame in [('ret', ret), ('vol', vol), ('volume_change', vol_change), ('drawdown', drawdown), ('momentum', momentum)]:
        features[name] = frame.replace([np.inf, -np.inf], np.nan)
    joined = pd.concat(features, axis=1).dropna(how='any')
    joined.to_parquet(DATA_DIR / 'features.parquet')
    ret.loc[joined.index].to_parquet(DATA_DIR / 'returns.parquet')
    return joined, ret.loc[joined.index]


def make_arrays(features, returns):
    dates = features.index
    n = len(ASSETS)
    f = 5
    x = np.stack([features[k].values.reshape(len(dates), n) for k in ['ret','vol','volume_change','drawdown','momentum']], axis=-1)
    y = returns.shift(-HORIZON).reindex(dates).values
    valid = np.isfinite(x).all(axis=(1,2)) & np.isfinite(y).all(axis=1)
    x = x[valid]
    y = y[valid]
    dates = dates[valid]
    samples = []
    targets = []
    shock = []
    sample_dates = []
    for i in range(LOOKBACK - 1, len(x)):
        samples.append(x[i-LOOKBACK+1:i+1])
        targets.append(y[i])
        shock.append(x[i-LOOKBACK+1:i+1,:,0])
        sample_dates.append(dates[i])
    return np.asarray(samples, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(shock, dtype=np.float32), pd.DatetimeIndex(sample_dates)


def temporal_split(x, y, u, dates):
    n = len(x)
    train_end = int(n * TRAIN_FRAC)
    valid_end = int(n * (TRAIN_FRAC + VALID_FRAC))
    train = slice(0, train_end)
    valid = slice(train_end + PURGE, valid_end)
    test = slice(valid_end + PURGE, n)
    return {'train': (x[train], y[train], u[train], dates[train]), 'valid': (x[valid], y[valid], u[valid], dates[valid]), 'test': (x[test], y[test], u[test], dates[test])}


def scale_data(splits):
    scaler = StandardScaler()
    train_x = splits['train'][0]
    scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
    result = {}
    for key, (x, y, u, dates) in splits.items():
        xs = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
        us = xs[:,:,:,0].astype(np.float32)
        result[key] = (xs, y.astype(np.float32), us, dates)
    with open(OUT_DIR / 'feature_scaler.json', 'w') as f:
        json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}, f)
    return result


class HysteresisNet(nn.Module):
    def __init__(self, n_assets, n_features=5, latent=24, memory=16):
        super().__init__()
        self.n_assets = n_assets
        self.n_features = n_features
        self.latent = latent
        self.memory = memory
        self.encoder = nn.Sequential(nn.Linear(n_features, 32), nn.GELU(), nn.Linear(32, latent), nn.Tanh())
        self.alpha = nn.Sequential(nn.Linear(latent + memory, 32), nn.GELU(), nn.Linear(32, memory), nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(latent + memory, 32), nn.GELU(), nn.Linear(32, memory), nn.Sigmoid())
        self.drift = nn.Sequential(nn.Linear(latent + memory, 48), nn.GELU(), nn.Linear(48, latent), nn.Tanh())
        self.graph_q = nn.Linear(latent + memory, 16, bias=False)
        self.graph_k = nn.Linear(latent + memory, 16, bias=False)
        self.noise = nn.Sequential(nn.Linear(latent + memory, 32), nn.GELU(), nn.Linear(32, latent), nn.Softplus())
        self.head = nn.Sequential(nn.Linear(latent + memory, 48), nn.GELU(), nn.Linear(48, 3))
        self.decoder = nn.Sequential(nn.Linear(latent, 32), nn.GELU(), nn.Linear(32, n_features))

    def forward(self, x, shock):
        b, t, n, f = x.shape
        h = self.encoder(x)
        memory_state = torch.zeros(b, n, self.memory, device=x.device)
        graph_last = None
        z_last = h[:, -1]
        for j in range(t):
            z = h[:, j]
            combined = torch.cat([z, memory_state], dim=-1)
            a = self.alpha(combined)
            beta = 0.03 + 0.97 * self.beta(combined)
            memory_state = memory_state + a * shock[:, j].unsqueeze(-1) - beta * memory_state
            q = self.graph_q(combined)
            k = self.graph_k(combined)
            logits = torch.einsum('bif,bjf->bij', q, k) / math.sqrt(q.shape[-1])
            graph = torch.softmax(logits, dim=-1)
            graph_last = graph
            propagated = torch.einsum('bij,bj->bi', graph, shock[:, j])
            drift = self.drift(torch.cat([z, memory_state], dim=-1))
            scale = self.noise(torch.cat([z, memory_state], dim=-1))
            z_last = z + 0.1 * (drift + scale * propagated.unsqueeze(-1))
        state = torch.cat([z_last, memory_state], dim=-1)
        out = self.head(state)
        reconstruction = self.decoder(h[:, -1])
        return out, reconstruction, graph_last, memory_state


def fit_hysteresis(train, valid, n_assets):
    xtr, ytr, utr, _ = train
    xva, yva, uva, _ = valid
    model = HysteresisNet(n_assets).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ds = TensorDataset(torch.tensor(xtr), torch.tensor(utr), torch.tensor(ytr))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    best = float('inf')
    best_state = None
    history = []
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb, ub, yb in loader:
            xb, ub, yb = xb.to(DEVICE), ub.to(DEVICE), yb.to(DEVICE)
            pred, recon, graph, memory = model(xb, ub)
            mean = pred[:,:,0]
            log_scale = torch.clamp(pred[:,:,1], -5, 3)
            tail_logit = pred[:,:,2]
            scale = torch.exp(log_scale)
            nll = 0.5 * ((yb - mean) / scale).pow(2) + log_scale
            recon_target = xb[:,-1]
            recon_loss = (recon - recon_target).pow(2).mean()
            sparsity = graph.abs().mean()
            memory_reg = memory.pow(2).mean()
            loss = nll.mean() + 0.05 * recon_loss + 0.01 * sparsity + 0.001 * memory_reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(xva).to(DEVICE)
            uv = torch.tensor(uva).to(DEVICE)
            yv = torch.tensor(yva).to(DEVICE)
            pv, rv, gv, mv = model(xv, uv)
            val = (0.5 * ((yv - pv[:,:,0]) / torch.exp(torch.clamp(pv[:,:,1], -5, 3))).pow(2) + torch.clamp(pv[:,:,1], -5, 3)).mean().item()
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'valid_nll': val})
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT_DIR / 'hysteresis_training.csv', index=False)
    return model


def predict_hysteresis(model, split):
    x, y, u, dates = split
    model.eval()
    preds = []
    graphs = []
    memories = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            xb = torch.tensor(x[i:i+BATCH]).to(DEVICE)
            ub = torch.tensor(u[i:i+BATCH]).to(DEVICE)
            out, recon, graph, memory = model(xb, ub)
            preds.append(out.cpu().numpy())
            graphs.append(graph.cpu().numpy())
            memories.append(memory.cpu().numpy())
    return np.concatenate(preds), np.concatenate(graphs), np.concatenate(memories)


def evaluate_predictions(y, pred, name):
    mean = pred if pred.ndim == 2 else pred[:,:,0]
    flat_y = y.reshape(-1)
    flat_p = mean.reshape(-1)
    ic = np.corrcoef(flat_y, flat_p)[0,1] if np.std(flat_y) > 0 and np.std(flat_p) > 0 else np.nan
    direction = float(np.mean(np.sign(flat_y) == np.sign(flat_p)))
    rmse = math.sqrt(mean_squared_error(flat_y, flat_p))
    pnl = flat_p * np.sign(flat_p)
    equity = np.cumprod(1 + np.clip(pnl, -0.25, 0.25))
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    return {'model': name, 'rank_ic': float(ic), 'directional_accuracy': direction, 'rmse': rmse, 'mean_signal_return': float(np.mean(pnl)), 'cumulative_signal_growth': float(equity[-1] - 1), 'max_drawdown': float(dd.min())}


def make_baselines(train, valid, test):
    xtr, ytr, _, _ = train
    xva, yva, _, _ = valid
    xte, yte, _, dte = test
    n_assets = ytr.shape[1]
    feat_tr = xtr[:,-1].reshape(len(xtr), -1)
    feat_va = xva[:,-1].reshape(len(xva), -1)
    feat_te = xte[:,-1].reshape(len(xte), -1)
    ytr_flat = ytr
    models = {}
    ridge = Ridge(alpha=10.0)
    ridge.fit(feat_tr, ytr_flat)
    models['ridge'] = ridge.predict(feat_te)
    rf = RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf.fit(feat_tr, ytr_flat)
    models['random_forest'] = rf.predict(feat_te)
    models['persistence'] = xte[:,-1,:,0]
    momentum = xte[:,-1,:,4]
    models['momentum'] = momentum
    return models, yte, dte


def plot_results(results, predictions, ytest, dates, graphs, memories):
    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT_DIR / 'metrics.csv', index=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=rdf, x='model', y='rank_ic')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rank_ic_comparison.png', dpi=180)
    plt.close()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=rdf, x='model', y='directional_accuracy')
    plt.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'directional_accuracy_comparison.png', dpi=180)
    plt.close()
    plt.figure(figsize=(12, 6))
    for name, p in predictions.items():
        signal = p if p.ndim == 2 else p[:,:,0]
        daily = np.mean(signal * ytest, axis=1)
        curve = np.cumprod(1 + np.clip(daily, -0.05, 0.05))
        plt.plot(dates, curve, label=name)
    plt.legend()
    plt.title('Out-of-sample signal equity curves')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'equity_curves.png', dpi=180)
    plt.close()
    final_graph = np.mean(graphs[-min(100, len(graphs)):], axis=0)
    plt.figure(figsize=(9, 8))
    sns.heatmap(final_graph, cmap='coolwarm', center=0)
    plt.title('Average learned shock propagation graph')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'learned_propagation_graph.png', dpi=180)
    plt.close()
    memory_norm = np.linalg.norm(memories, axis=-1).mean(axis=1)
    plt.figure(figsize=(12, 4))
    plt.plot(dates, memory_norm)
    plt.title('Average HYSTERESIS memory magnitude')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'memory_magnitude.png', dpi=180)
    plt.close()


def main():
    features, returns = load_or_build_features()
    x, y, u, dates = make_arrays(features, returns)
    splits = scale_data(temporal_split(x, y, u, dates))
    model = fit_hysteresis(splits['train'], splits['valid'], len(ASSETS))
    hpred, graphs, memories = predict_hysteresis(model, splits['test'])
    baselines, ytest, test_dates = make_baselines(splits['train'], splits['valid'], splits['test'])
    predictions = {'hysteresis': hpred}
    predictions.update(baselines)
    results = [evaluate_predictions(ytest, hpred, 'hysteresis')]
    for name, pred in baselines.items():
        results.append(evaluate_predictions(ytest, pred, name))
    plot_results(results, predictions, ytest, test_dates, graphs, memories)
    metadata = {'assets': ASSETS, 'asset_groups': TICKERS, 'n_assets': len(ASSETS), 'device': DEVICE, 'lookback': LOOKBACK, 'horizon': HORIZON, 'train_frac': TRAIN_FRAC, 'valid_frac': VALID_FRAC, 'purge': PURGE, 'train_samples': len(splits['train'][0]), 'valid_samples': len(splits['valid'][0]), 'test_samples': len(splits['test'][0]), 'date_start': str(dates[0]), 'date_end': str(dates[-1])}
    with open(OUT_DIR / 'run_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps({'metrics': results, 'metadata': metadata}, indent=2))


if __name__ == '__main__':
    main()

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import torch
import sys
base = sys.modules[__name__]

ROOT = Path('/content/hysteresis_research') if Path('/content').exists() else Path.cwd() / 'hysteresis_research'
OUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
COST_BPS = 5.0


def rank_ic_by_date(y, pred):
    values = []
    for t in range(y.shape[0]):
        mask = np.isfinite(y[t]) & np.isfinite(pred[t])
        if mask.sum() >= 3:
            values.append(spearmanr(pred[t, mask], y[t, mask]).statistic)
    return float(np.nanmean(values)) if values else np.nan


def regime_labels(test_u, train_u):
    train_state = np.mean(np.abs(train_u[:, -1]), axis=1)
    test_state = np.mean(np.abs(test_u[:, -1]), axis=1)
    q75, q90 = np.quantile(train_state, [0.75, 0.90])
    labels = np.full(len(test_state), 'calm', dtype=object)
    labels[test_state >= q75] = 'stress'
    labels[test_state >= q90] = 'shock'
    for i in range(1, len(test_state)):
        if train_state[-1] if i == 0 else test_state[i - 1] >= q75:
            if test_state[i] < q75:
                labels[i] = 'recovery'
    return labels, {'stress_threshold': float(q75), 'shock_threshold': float(q90)}


def positions(pred):
    signal = pred if pred.ndim == 2 else pred[:, :, 0]
    scale = np.nanmedian(np.abs(signal), axis=1, keepdims=True) + 1e-6
    raw = np.clip(signal / (3.0 * scale), -1.0, 1.0)
    denom = np.sum(np.abs(raw), axis=1, keepdims=True) + 1e-8
    return raw / denom


def portfolio_returns(pred, y):
    w = positions(pred)
    turnover = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
    gross = np.sum(w[1:] * y[1:], axis=1)
    costs = turnover * COST_BPS / 10000.0
    net = gross - costs
    return net, gross, turnover


def score_segment(y, pred, labels, name, segment):
    mask = labels == segment
    if mask.sum() == 0:
        return {'model': name, 'regime': segment, 'dates': 0, 'rank_ic': np.nan, 'directional_accuracy': np.nan, 'rmse': np.nan, 'net_sharpe': np.nan, 'net_mean_return': np.nan, 'net_cumulative_return': np.nan, 'turnover': np.nan}
    yy = y[mask]
    pp = pred[mask] if pred.ndim == 2 else pred[mask, :, 0]
    signal = pp
    da = float(np.mean(np.sign(signal) == np.sign(yy)))
    rmse = float(np.sqrt(np.mean((signal - yy) ** 2)))
    ic = rank_ic_by_date(yy, signal)
    net, gross, turnover = portfolio_returns(signal, yy)
    if len(net) >= 2 and np.std(net) > 0:
        sharpe = float(np.sqrt(252) * np.mean(net) / np.std(net, ddof=1))
    else:
        sharpe = np.nan
    cumulative = float(np.prod(1.0 + np.clip(net, -0.25, 0.25)) - 1.0) if len(net) else np.nan
    return {'model': name, 'regime': segment, 'dates': int(mask.sum()), 'rank_ic': ic, 'directional_accuracy': da, 'rmse': rmse, 'net_sharpe': sharpe, 'net_mean_return': float(np.mean(net)) if len(net) else np.nan, 'net_cumulative_return': cumulative, 'turnover': float(np.mean(turnover)) if len(turnover) else np.nan}


def all_scores(y, pred, labels, name):
    rows = [score_segment(y, pred, labels, name, regime) for regime in ['calm', 'shock', 'stress', 'recovery']]
    rows.append(score_segment(y, pred, np.ones(len(labels), dtype=object), name, 'all'))
    return rows


def corrected_baselines(train, test, feature_mean, feature_scale):
    xtr, ytr, utr, dtr = train
    xte, yte, ute, dte = test
    raw_ret = xte[:, -1, :, 0] * feature_scale[0] + feature_mean[0]
    raw_mom = xte[:, -1, :, 4] * feature_scale[4] + feature_mean[4]
    feat_tr = xtr[:, -1].reshape(len(xtr), -1)
    feat_te = xte[:, -1].reshape(len(xte), -1)
    ridge = base.Ridge(alpha=10.0)
    ridge.fit(feat_tr, ytr)
    rf = base.RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=5, random_state=base.SEED, n_jobs=-1)
    rf.fit(feat_tr, ytr)
    return {'ridge': ridge.predict(feat_te), 'random_forest': rf.predict(feat_te), 'persistence': raw_ret, 'momentum': raw_mom}


def regime_plots(df):
    for metric in ['rank_ic', 'rmse', 'net_sharpe', 'directional_accuracy']:
        plt.figure(figsize=(11, 6))
        sns.barplot(data=df[df.regime != 'all'], x='regime', y=metric, hue='model')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f'regime_{metric}.png', dpi=180)
        plt.close()
    pivot = df[df.regime != 'all'].pivot(index='model', columns='regime', values='rank_ic')
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', center=0, cmap='coolwarm')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'regime_rank_ic_heatmap.png', dpi=180)
    plt.close()


def main():
    features, returns = base.load_or_build_features()
    x, y, u, dates = base.make_arrays(features, returns)
    raw_train_end = int(len(x) * base.TRAIN_FRAC)
    feature_mean = x[:raw_train_end].reshape(-1, x.shape[-1]).mean(axis=0)
    feature_scale = x[:raw_train_end].reshape(-1, x.shape[-1]).std(axis=0) + 1e-8
    splits = base.scale_data(base.temporal_split(x, y, u, dates))
    model = base.fit_hysteresis(splits['train'], splits['valid'], len(base.ASSETS))
    hpred, graphs, memories = base.predict_hysteresis(model, splits['test'])
    baselines = corrected_baselines(splits['train'], splits['test'], feature_mean, feature_scale)
    ytest = splits['test'][1]
    test_u = splits['test'][2]
    train_u = splits['train'][2]
    labels, thresholds = regime_labels(test_u, train_u)
    predictions = {'hysteresis': hpred[:, :, 0]}
    predictions.update(baselines)
    rows = []
    for name, pred in predictions.items():
        rows.extend(all_scores(ytest, pred, labels, name))
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / 'regime_metrics.csv', index=False)
    pd.DataFrame({'date': splits['test'][3], 'regime': labels}).to_csv(OUT_DIR / 'test_regimes.csv', index=False)
    with open(OUT_DIR / 'regime_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    regime_plots(result)
    memory_norm = np.linalg.norm(memories, axis=-1).mean(axis=1)
    plt.figure(figsize=(12, 4))
    plt.plot(splits['test'][3], memory_norm)
    for regime, color in [('shock', 'red'), ('stress', 'orange'), ('recovery', 'green')]:
        idx = np.where(labels == regime)[0]
        if len(idx):
            plt.scatter(splits['test'][3][idx], memory_norm[idx], s=8, label=regime, color=color)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'memory_by_regime.png', dpi=180)
    plt.close()
    print(result.to_string(index=False))
    print(json.dumps({'regime_thresholds': thresholds, 'counts': pd.Series(labels).value_counts().to_dict()}, indent=2))


if __name__ == '__main__':
    main()
