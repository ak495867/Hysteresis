# HYSTERESIS — A Path-Dependent Market Response Model
### A Beginner-Friendly Mathematical & Implementation Guide

> **Who this is for:** you don't need a background in stochastic calculus, graph theory, or physics to follow this. Every symbol is defined the first time it appears, every formula is followed by a plain-English translation, and the sections build in order. If you can read a flowchart and you know what a "moving average" is, you can follow this document all the way through.

---

## 1. What HYSTERESIS Is Trying to Do

Picture two moments in the market that look *identical* on paper: same price, same volatility, same trading volume. A typical model — one that only looks at "the current numbers" — would predict the exact same future for both.

**HYSTERESIS starts from the belief that this is wrong**, and that the two moments can behave very differently depending on *how the market got there*:

> **Hypothesis:** the market's current state is not enough to predict what happens next. You also need to know the *path* it took to arrive at that state — did volatility build up slowly, or did it come from a sudden shock the market is still recovering from? Two markets with the same "snapshot" but different histories will, on average, behave differently in the near future.

The word "hysteresis" comes from physics, where it describes systems that "remember" what happened to them — like a metal spring that doesn't spring back to *exactly* its original shape after being stretched too far, because the stretching left a lasting mark. HYSTERESIS proposes that liquidity, volatility, and risk in markets behave the same way: **a shock leaves a memory trace that fades at its own pace, and that trace changes how the next shock will play out.**

Rather than folding "history" into one big hidden number the way most sequence models do, HYSTERESIS deliberately separates the market into five named, individually-inspectable pieces (Section 2), so you can test each idea on its own — including testing whether "memory" helps at all.

---

## 2. The Five Pieces of the Model (In Plain English First)

Before any math, here's the mental model. Think of a market like a lake that occasionally gets hit by rocks (shocks):

| Piece | Plain-English role |
| --- | --- |
| **Instantaneous state** | "What does the lake's surface look like right now?" — current price, volatility, liquidity, etc. |
| **Shock field** | "Is a rock hitting the water right now?" — new incoming pressure: a macro announcement, a big sell order, a surprise earnings report |
| **Memory state** | "How rippled is the water from *previous* rocks that haven't fully settled yet?" — a compressed summary of recent shock-and-recovery history |
| **Response operator** | "If a rock lands *here*, how far and how strongly do the ripples spread to other parts of the lake?" — how a shock in one asset propagates to others, and this can change depending on current conditions |
| **Transition hazard** | "How likely is the lake to suddenly change character altogether?" — e.g., moving from "calm" to "storm" |

The model's central, genuinely new object is the **response operator**: instead of just predicting "what will the price do," it separately predicts "how will a shock spread through the system, given what's happened recently" — a testable, inspectable intermediate quantity.

---

## 3. The Cast of Characters (Notation Glossary)

Read this once, then use it as a lookup table.

| Symbol | Plain-English meaning |
| --- | --- |
| $t$ | A point in time |
| $X_t$ | The full observed market snapshot at time $t$ (returns, order flow, volatility, liquidity, macro/news signals, cross-asset relationships — see Section 4) |
| $z_t$ | The **instantaneous state**: a compressed numeric summary of "what the market looks like right now" |
| $m_t$ | The **memory state**: a compressed numeric summary of "how much unresolved shock/recovery history is still lingering" |
| $u_t$ | The **shock vector**: how much new external pressure is hitting the system right now |
| $E_\theta(\cdot)$ | The **encoder** network (parameters $\theta$) that turns raw recent data into $z_t$ |
| $\alpha(\cdot), \beta(\cdot)$ | Learned functions controlling how fast memory builds up ($\alpha$) and fades ($\beta$) |
| $F_\phi(\cdot)$ | A network describing how the state drifts on its own, with no shocks |
| $B_\phi(\cdot)$ | The **response operator**: describes how a shock propagates through the system, and can depend on both current state and memory |
| $A_\phi$ | A learned matrix representing a directed "who-affects-whom" graph between assets |
| $D_\phi$ | A normalization matrix used to keep $A_\phi$ numerically well-behaved (a standard graph-processing trick) |
| $\Sigma_\phi(\cdot)$ | A learned function describing the size of ordinary random noise in the system |
| $\varepsilon_t$ | Ordinary random noise (unpredictable "wiggle") |
| $p_\psi(\cdot)$ | A probability distribution predicted by a network with parameters $\psi$, rather than a single number |
| $Y_{t+h}$ | The bundle of things we're trying to predict $h$ steps into the future (return, volatility, tail risk, recovery time) |
| $\mathcal{L}$ | A **loss function** — a number the model tries to shrink during training |
| $\lambda_i$ | Hand-set weights balancing the different pieces of the loss |
| $\|\cdot\|^2$ | Squared length of a vector — "add up the squares of every entry"; used to measure "how far apart are these two things" |
| $\|\cdot\|_1$ | The sum of absolute values of a vector's entries — used to encourage most entries to be exactly zero ("sparsity") |
| $\|\cdot\|_F$ | The Frobenius norm — the matrix version of "how big is this, overall," used the same way as $\|\cdot\|^2$ but for whole matrices |

---

## 4. Step 1 — What Goes Into the Model

At each time $t$, we assemble a market observation vector made of several named blocks:

$$
X_t = [\,R_t,\ Q_t,\ V_t,\ L_t,\ M_t,\ G_t\,],
$$

where:

- $R_t$ — returns (how much did prices move)
- $Q_t$ — order-flow imbalance (are more people buying or selling right now)
- $V_t$ — volatility variables (how jumpy has price been)
- $L_t$ — liquidity variables (how easy is it to trade without moving the price)
- $M_t$ — macro/news variables (interest rates, headlines, economic releases)
- $G_t$ — cross-asset relationship variables (how are related assets behaving)

This is simply "collect everything relevant, in one labeled bundle" — no modeling happens yet.

---

## 5. Step 2 — Compressing the Present Into a State

A neural network encoder looks at a recent window of data and produces the instantaneous state:

$$
z_t = E_\theta\big(X_{t-k:t}\big).
$$

In words: *"look at the last $k$ time steps of raw data, and summarize it into a compact instantaneous state $z_t$."* This is the same idea as an encoder in many other sequence models — nothing novel happens yet here. The novelty starts in the next step.

---

## 6. Step 3 — Giving the Market a Separate Memory

This is the model's central idea. Rather than baking "history" invisibly into $z_t$ the way a typical recurrent model would, HYSTERESIS keeps a **separate, dedicated memory state** $m_t$, whose whole job is to track unresolved shock/recovery history:

$$
\frac{dm_t}{dt} = \alpha(z_t, m_t)\, u_t \;-\; \beta(z_t, m_t)\, m_t.
$$

Reading the right-hand side term by term:

- $\alpha(z_t, m_t)\, u_t$ — the **build-up** term. When a shock $u_t$ arrives, this term adds to the memory, and *how much* it adds depends on the current state and existing memory (a market that's already stressed might absorb — or amplify — a new shock differently than a calm one).
- $-\beta(z_t, m_t)\, m_t$ — the **decay/recovery** term. This subtracts from memory over time, representing the market gradually "forgetting" past shocks — but crucially, $\beta$ (the recovery speed) is itself learned and state-dependent, **not** a fixed number.

**Why this matters:** in a plain model, you might assume memory always decays like a simple half-life (e.g., "shocks fade by half every 3 days, always"). HYSTERESIS explicitly rejects that assumption — it lets the model learn that, for example, liquidity memory during a calm period might fade quickly, while memory left over from a full-blown crisis might linger much longer. This is the mathematical expression of "hysteresis": the recovery rate itself depends on what state the system is in.

---

## 7. Step 4 — How a Shock Spreads: The Response Operator

Now we describe how the instantaneous state itself evolves, combining ordinary drift, shock propagation, and noise:

$$
\frac{dz_t}{dt} = F_\phi(z_t) \;+\; B_\phi(z_t, m_t)\, u_t \;+\; \Sigma_\phi(z_t, m_t)\, \varepsilon_t.
$$

Term by term:

- $F_\phi(z_t)$ — the **self-drift**: how the state would evolve on its own, with no new shocks, similar to how a ball rolls toward the bottom of a bowl.
- $B_\phi(z_t, m_t)\, u_t$ — the **response operator** acting on the current shock. This is the star of the model: $B_\phi$ describes *how a shock spreads*, and — critically — it's allowed to depend on both the current state $z_t$ *and* the memory $m_t$. The same-sized shock can propagate differently depending on whether the market is calm or already carrying leftover stress from a recent event.
- $\Sigma_\phi(z_t, m_t)\, \varepsilon_t$ — ordinary unpredictable noise, whose *size* is also allowed to depend on state and memory (markets are noisier in some conditions than others).

### 7.1 Making the response operator interpretable: a shock-propagation graph

For markets with many assets, $B_\phi$ is structured as a **directed graph** — a "who affects whom, and how strongly" map:

$$
B_\phi(z_t, m_t) = D_\phi(z_t,m_t)^{-1/2}\, A_\phi(z_t,m_t)\, D_\phi(z_t,m_t)^{-1/2}.
$$

- $A_\phi(z_t, m_t)$ — a matrix where entry $(i,j)$ represents "how strongly does a shock in asset $i$ push on asset $j$, right now." Unlike a plain correlation matrix, this is **directed** (asset $i$ can affect asset $j$ more than $j$ affects $i$) and **changes over time** with the current state and memory.
- $D_\phi(z_t, m_t)^{-1/2}(\cdot) D_\phi(z_t,m_t)^{-1/2}$ — a standard normalization "sandwich" (the same trick used in graph neural networks) that keeps the numbers well-scaled so the model doesn't blow up or shrink to nothing as shocks propagate through many connected assets.

**Why this is more than a black box:** because $A_\phi$ is an explicit matrix, you can print it out, visualize it as a graph, and check things like: "does the model think shocks flow from big banks to smaller regional ones, and does that match what we already know about the real financial system?"

---

## 8. Step 5 — Predicting a Whole Distribution, Not Just One Number

Instead of outputting a single predicted return, HYSTERESIS predicts a full **probability distribution** over several related outcomes:

$$
p_\psi\big(Y_{t+h} \mid z_t, m_t, \hat{z}_{t:t+h}\big),
$$

where the target bundle is

$$
Y_{t+h} = \begin{bmatrix} r_{t:t+h} \\ \sigma_{t:t+h} \\ P(r_{t:t+h} < -k) \\ \text{liquidity recovery time} \end{bmatrix}.
$$

In plain terms, the model is asked to predict, simultaneously:

- $r_{t:t+h}$ — the expected return over the next $h$ steps
- $\sigma_{t:t+h}$ — the expected volatility over that window
- $P(r_{t:t+h} < -k)$ — the probability of a large loss (worse than some threshold $-k$) — i.e. tail risk
- **liquidity recovery time** — how long it will take liquidity to return to normal after any current disturbance

That last item is included *on purpose*: it directly tests whether the model has actually learned something about *recovery as a process*, rather than just treating volatility as an isolated, memoryless number.

---

## 9. Step 6 — The Training Objective, Piece by Piece

Training combines six separate loss terms, each checking a different part of the story:

$$
\mathcal{L} = \mathcal{L}_{\text{forecast}} + \lambda_1 \mathcal{L}_{\text{state}} + \lambda_2 \mathcal{L}_{\text{propagation}} + \lambda_3 \mathcal{L}_{\text{memory}} + \lambda_4 \mathcal{L}_{\text{sparsity}} + \lambda_5 \mathcal{L}_{\text{calibration}}.
$$

Here, each $\lambda_i$ is a hand-set dial controlling how much weight that piece gets relative to the others.

### 9.1 Forecast loss — "did the distribution match reality?"

$$
\mathcal{L}_{\text{forecast}} = -\log p_\psi\big(Y_{t+h} \mid z_t, m_t\big).
$$

This is called a **negative log-likelihood loss**: it rewards the model for assigning high probability to what *actually* happened, and penalizes it for being confidently wrong. Because we're predicting a full distribution rather than one number, this loss naturally also punishes overconfident or miscalibrated forecasts.

### 9.2 State reconstruction loss — "does $z_t$ still mean something real?"

$$
\mathcal{L}_{\text{state}} = \big\| D(z_t) - X_t \big\|^2.
$$

A decoder $D(\cdot)$ tries to reconstruct real, observable quantities (like spread, depth, volatility) from $z_t$. This is ordinary mean-squared error — squaring the gap between the reconstruction and the real value — and it stops $z_t$ from drifting into an arbitrary internal code that only "cheats" on the forecast without meaning anything real.

### 9.3 Propagation loss — "did the response operator actually get it right?"

$$
\mathcal{L}_{\text{propagation}} = \Big\| \widehat{\Delta X}_{t:t+H}^{\text{shock}} - \Delta X_{t:t+H}^{\text{observed}} \Big\|^2.
$$

Whenever there's a clearly identifiable real-world shock (a macro announcement, an index rebalance, an earnings surprise), this loss compares the model's *predicted* cross-asset ripple effect against what *actually* happened. This keeps $B_\phi$ honest and testable, instead of letting it become an uninterpretable internal matrix that only helps the loss number without meaning anything.

### 9.4 Memory consistency loss — "does memory actually explain different outcomes?"

$$
\mathcal{L}_{\text{memory}} = \max\Big(0,\ \delta - \operatorname{dist}(m_a, m_b)\Big).
$$

This is a **contrastive loss**. Take two moments, $a$ and $b$, that had a *similar current state* but led to *different realized outcomes* (say, one recovered quickly and one didn't). This loss says: *if the outcomes were meaningfully different, the memory states $m_a$ and $m_b$ had better be at least $\delta$ apart too* — otherwise it applies a penalty. In words: "memory should only be allowed to matter, and be different, when it's actually explaining something the current state alone can't."

### 9.5 Sparsity and stability loss — "keep the shock-propagation graph simple and steady"

$$
\mathcal{L}_{\text{sparsity}} = \|A_\phi\|_1 + \gamma \|A_\phi - A_\phi^{\text{EMA}}\|_F^2.
$$

- $\|A_\phi\|_1$ — encourages most entries of the propagation graph to be exactly zero, i.e., "most assets don't directly affect most other assets" — a sparse, interpretable graph rather than a dense mess of weak connections.
- $\|A_\phi - A_\phi^{\text{EMA}}\|_F^2$ — compares the current graph to its own recent exponential moving average ($A_\phi^{\text{EMA}}$), penalizing the graph for flickering wildly from one time step to the next. $\gamma$ controls how strongly we enforce this stability.

### 9.6 Calibration loss — "are the probabilities themselves trustworthy?"

Uses standard **proper scoring rules** — negative log-likelihood, the Continuous Ranked Probability Score (CRPS), and expected calibration error — to directly reward the model for producing tail-risk probabilities that are *actually* right roughly as often as claimed (e.g., events the model calls "10% likely" really do happen about 10% of the time).

---

## 10. Putting It All Together: The Full Pipeline

$$
X_t \xrightarrow{E_\theta} z_t
$$
$$
u_t,\ z_t,\ m_t \;\xrightarrow{\alpha,\ \beta}\; m_{t+\Delta} \quad \text{(memory update)}
$$
$$
z_t,\ m_t,\ u_t \;\xrightarrow{F_\phi,\ B_\phi,\ \Sigma_\phi}\; z_{t+\Delta} \quad \text{(state update)}
$$
$$
z_t,\ m_t,\ \hat z_{t:t+h} \;\xrightarrow{p_\psi}\; Y_{t+h} \quad \text{(distributional forecast)}
$$

| Stage | What goes in | What comes out | Plain-English role |
| --- | --- | --- | --- |
| Encoder | recent raw market data | $z_t$ | "Summarize what the market looks like right now" |
| Memory update | shock $u_t$, state $z_t$, old memory $m_t$ | new $m_t$ | "Track how much unresolved shock/recovery history is still lingering" |
| Response operator $B_\phi$ | state, memory | propagation graph $A_\phi$ | "Model how a shock right now would ripple across assets" |
| State dynamics | $z_t$, $m_t$, $u_t$, noise | future $z_{t+\Delta}$ | "Simulate the market's evolving condition forward in time" |
| Distributional head $p_\psi$ | $z_t$, $m_t$, simulated future states | return, volatility, tail-risk, recovery-time distribution | "Turn everything into a full, calibrated forecast" |

---

## 11. Why This Is a Genuinely Distinct Frontier Idea

| A generic "latent dynamics" proposal | HYSTERESIS |
| --- | --- |
| Learns one latent state and evolves it | Learns state *and* a separate, dedicated memory of shock/recovery history |
| History is buried invisibly inside hidden activations | History has its own named variable ($m_t$) with a learned, state-dependent recovery rate |
| Usually predicts one target (return or volatility) | Predicts return, volatility, tail risk, *and* liquidity recovery time together |
| Cross-asset interaction is often static or implicit | Shock propagation is an explicit, directed, sparse, state-dependent graph you can inspect |
| Interpretability is usually an after-the-fact add-on | Memory levels, propagation edges, and recovery rates are native, inspectable model outputs from the start |

The underlying economic idea is simple to state even without any of the math: **liquidity and risk have memory, and how a shock plays out depends on the state the market is already in when it arrives.** Everything above is just a careful, testable mathematical way of expressing that one sentence.

---

## 12. How You'd Actually Test the Hypothesis

The model makes several specific, falsifiable claims. Here's how each one would be tested:

| Claim | How to test it |
| --- | --- |
| **H1: Path memory adds real predictive value** | Compare the full model against an identical version with $m_t$ removed or randomized — if performance doesn't drop, memory isn't actually helping |
| **H2: State-dependent propagation beats a fixed correlation matrix** | Compare the learned, changing $A_\phi$ against a plain static correlation matrix and a simple rolling-correlation baseline |
| **H3: Memory matters most during regime transitions** | Evaluate performance separately in calm periods, stressed periods, recovery periods, and transition periods |
| **H4: Modeling recovery improves tail-risk forecasts** | Compare calibration and CRPS scores for downside-risk and recovery-time predictions, with and without the memory/recovery machinery |
| **H5: The learned response operator generalizes** | Train on one group of assets (e.g., one sector or exchange) and test on a different one, to see if the learned propagation patterns still hold up |

### A disciplined evaluation protocol

- **Strict chronological splitting:** train on an early period, validate on a later period, test only once on a final, truly unseen period — with a purge/embargo gap wherever labels could overlap in time, to avoid leaking future information into training.
- **Fair baseline ladder**, using the same features and horizon throughout: linear/ridge regression, XGBoost/LightGBM, LSTM/TCN, a Transformer, a plain latent ODE/SDE (no memory), and a graph-based temporal model.
- **Key ablations:** remove the memory state $m_t$; keep memory but remove the dynamic graph; keep the dynamic graph but remove the propagation loss; shuffle the shock timestamps; replace the learned recovery rate with a fixed one. Each of these should be checked, one at a time, to see exactly which piece is doing the work.
- **Metrics:** rank information coefficient, directional accuracy, negative log-likelihood, CRPS, tail-event calibration, regime-transition detection accuracy, turnover-adjusted returns, and how performance degrades as transaction costs increase. A full trading backtest comes last, as the final real-world sanity check — not as the primary definition of whether the model "worked."

---

## 13. A Practical First Version (V1)

You don't need a giant model to start testing this hypothesis. A modest, auditable prototype:

| Choice | V1 setting |
| --- | --- |
| Assets | 20–100 liquid equities, ETFs, or futures sharing the same timestamps |
| Data | Returns, volume, spread, realized volatility, order-flow proxies, sector/macro variables |
| Horizons | Tested separately at 5-minute, 1-hour, and 1-day scales |
| Encoder | A small temporal convolution network or compact Transformer |
| Memory | A 16–64 dimensional gated state |
| Propagation graph | Sparse, directed, over assets or sectors |
| Dynamics | Start with a deterministic Neural ODE; only add stochastic (SDE) noise once the simpler version is stable |
| Outputs | Return distribution, volatility, downside probability, recovery-time estimate |
| Evaluation | Walk-forward validation, purged splits, and an explicit leakage audit |

---

## 14. Honest Limitations

- **Identifying "shocks" cleanly is itself hard.** The propagation loss (Section 9.3) depends on being able to point to specific, identifiable shock events — in practice, many real market moves don't come with a clean, unambiguous trigger.
- **More moving parts means more ways to overfit.** With six loss terms and five hyperparameters ($\lambda_1$–$\lambda_5$, plus $\gamma$), careful ablation (Section 12) is essential to confirm each piece is actually earning its place, rather than just adding flexibility that memorizes noise.
- **A decreasing training loss does not prove real predictive skill.** As with any model, everything here must be validated strictly out-of-sample, on unseen assets and unseen time periods, with realistic transaction costs included.
- **This is a research prototype, not a production trading system.** It has no live execution logic, no portfolio-level risk controls, and no formal position-sizing framework — those remain necessary, separate engineering layers on top of anything described here.
- **Novelty should be verified, not assumed.** As the original proposal itself notes, related ideas (neuro-symbolic latent SDEs, financial foundation models, non-linear market-impact models, and Koopman-operator approaches to causal discovery) already exist in the literature — a careful search is needed before claiming this exact combination is unprecedented.

---

## 15. Quick Reference: All Formulas in One Place

| # | Formula | Meaning in one line |
| --- | --- | --- |
| 1 | $X_t = [R_t, Q_t, V_t, L_t, M_t, G_t]$ | The full labeled market observation bundle |
| 2 | $z_t = E_\theta(X_{t-k:t})$ | Encoder: recent data → instantaneous state |
| 3 | $\frac{dm_t}{dt} = \alpha(z_t,m_t)u_t - \beta(z_t,m_t)m_t$ | Memory update: shocks build it up, learned state-dependent recovery decays it |
| 4 | $\frac{dz_t}{dt} = F_\phi(z_t) + B_\phi(z_t,m_t)u_t + \Sigma_\phi(z_t,m_t)\varepsilon_t$ | State dynamics: self-drift + shock propagation + noise |
| 5 | $B_\phi(z_t,m_t) = D_\phi^{-1/2}A_\phi D_\phi^{-1/2}$ | Response operator as a normalized directed propagation graph |
| 6 | $p_\psi(Y_{t+h}\mid z_t,m_t,\hat z_{t:t+h})$ | Full distributional forecast (not just a point estimate) |
| 7 | $\mathcal{L} = \mathcal{L}_{\text{forecast}} + \lambda_1\mathcal{L}_{\text{state}} + \lambda_2\mathcal{L}_{\text{propagation}} + \lambda_3\mathcal{L}_{\text{memory}} + \lambda_4\mathcal{L}_{\text{sparsity}} + \lambda_5\mathcal{L}_{\text{calibration}}$ | Full six-part training objective |
| 8 | $\mathcal{L}_{\text{forecast}} = -\log p_\psi(Y_{t+h}\mid z_t,m_t)$ | Reward correct, calibrated distributional predictions |
| 9 | $\mathcal{L}_{\text{state}} = \|D(z_t)-X_t\|^2$ | Keep the state faithful to real observables |
| 10 | $\mathcal{L}_{\text{propagation}} = \|\widehat{\Delta X}^{\text{shock}}-\Delta X^{\text{observed}}\|^2$ | Keep the response operator honest against real shock outcomes |
| 11 | $\mathcal{L}_{\text{memory}} = \max(0,\ \delta - \operatorname{dist}(m_a,m_b))$ | Memory should differ only when outcomes actually differ |
| 12 | $\mathcal{L}_{\text{sparsity}} = \|A_\phi\|_1 + \gamma\|A_\phi - A_\phi^{\text{EMA}}\|_F^2$ | Keep the propagation graph sparse and stable over time |

---

## References

1. [ARTEMIS: A Neuro-Symbolic Framework for Economically Constrained Market Dynamics](https://arxiv.org/html/2603.18107v1)
2. [FinCast: A Foundation Model for Financial Time-Series Forecasting](https://arxiv.org/html/2508.19609v1)
3. [A Fully Consistent, Minimal Model for Non-Linear Market Impact](https://www.cfm.com/wp-content/uploads/2022/12/38-2014-A-fully-consistent-minimal-model-for-non-linear-market-impact.pdf)
4. [Deep Koopman Operators for Causal Discovery](https://www.nature.com/articles/s42005-025-02426-1)