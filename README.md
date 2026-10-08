# Real-Time Customer Engagement Analytics using Big Data Analytics and AI

## Project Title

**Real-Time Customer Engagement Analytics using Big Data Analytics and AI**

## Problem Statement

Develop a real-time customer‑engagement analytics model using clickstream data, synthetic sentiment, recommender‑system logs and customer lifetime value (CLV).

## Reference Paper

Arshad et al. (2025), *Big data analytics and AI as success factors for online video streaming platforms*.

## Project Objective

Implement a research prototype that follows the **four‑stage framework**:

1. **Transformer Embeddings** – Learn vector representations of customers from their behavioral event sequences.
2. **Customer Segmentation** – Cluster customers using embeddings + CLV/CLV‑proxy.
3. **Dynamic Recommendation** – Generate personalized recommendations based on the embedding, segment, and recent context.
4. **Causal Analysis** – Estimate the impact of the recommendation via a simulated treatment‑control experiment.

The prototype respects research integrity: all synthetic data are clearly labeled, no fabricated real‑world results are claimed.

## Dataset

We use the **RetailRocket** recommender‑system dataset.  The required file is:
```
DAV_SDG/data/events.csv
```
It contains real click‑stream events (view, addtocart, transaction) with columns `timestamp, visitorid, event, itemid, transactionid`.

> **Dataset limitation** – RetailRocket does **not** contain explicit sentiment or monetary values.  Sentiment and CLV are generated as synthetic proxies.

## System Architecture

```
DAV_SDG/
├─ data/                     # Raw dataset (events.csv)
├─ models/                   # Trained model checkpoints
├─ outputs/                  # All generated CSVs & visualizations
├─ notebooks/                # Optional Jupyter notebooks for exploration
├─ src/                     
│   ├─ __init__.py
│   ├─ config.py            # Central configuration
│   ├─ preprocessing.py     # Data cleaning & feature engineering
│   ├─ transformer_embeddings.py
│   ├─ customer_segmentation.py
│   ├─ dynamic_recommendation.py
│   ├─ causal_analysis.py
│   └─ realtime_engine.py   # Real‑time simulation engine
├─ main.py                  # End‑to‑end pipeline orchestrator
├─ requirements.txt         # Python dependencies
├─ README.md                # This document
└─ .gitignore               # Ignored files
```

## Installation & Environment Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt
```

## Adding the Dataset

Place `events.csv` inside `DAV_SDG/data/`.  If the file is missing, the pipeline will abort with a clear message.

## How to Run the Prototype

```bash
python main.py
```
The script executes the complete pipeline (pre‑processing → embeddings → segmentation → recommendation → causal analysis → real‑time demo) and writes all results to `outputs/`.

## Module Overview

* `src/config.py` – Centralised configurable parameters (e.g., `MAX_USERS`, `EMBEDDING_DIM`).
* `src/preprocessing.py` – Loads `events.csv`, cleans data, creates user‑level features and saves intermediate CSVs.
* `src/transformer_embeddings.py` – Defines a lightweight PyTorch Transformer encoder, trains it on event sequences, and produces user embeddings.
* `src/customer_segmentation.py` – Performs K‑Means clustering on embeddings + CLV proxy, evaluates clusters, and saves segment labels.
* `src/dynamic_recommendation.py` – Generates top‑K personalized recommendations using a candidate‑generation + ranking pipeline.
* `src/causal_analysis.py` – Simulates a randomized treatment/control experiment and estimates the Average Treatment Effect (ATE).
* `src/realtime_engine.py` – Simulates online event processing; updates user state, re‑computes embedding/segment, and logs recommendations.
* `main.py` – Orchestrates all steps, handles missing data, prints key metrics and visualisations.

## Transformer Architecture

A small Transformer encoder (CPU‑friendly) with:
* Event, item and sentiment embeddings (dimension `EMBEDDING_DIM = 64`).
* Positional encoding.
* Two encoder layers.
* Mean‑pooling over the sequence to obtain a fixed‑size user vector.
Training uses a next‑event prediction objective on the click‑stream sequences.

## Synthetic Sentiment Proxy Methodology

Because RetailRocket lacks textual sentiment, we compute a **behavioral sentiment score** based on:
* Frequency of `view`, `addtocart`, `transaction` events.
* Ratio of purchases to views (cart conversion).
* Recency of activity.
The score is deterministic (fixed random seed) and stored in `outputs/synthetic_sentiment.csv` with columns `sentiment_score`, `sentiment_label` (`positive/neutral/negative`) and `sentiment_source = "synthetic_behavioral_proxy"`.

## CLV / CLV‑Proxy Methodology

We compute a **CLV proxy** from actual transaction counts and recency.  Monetary value is not available, so the proxy is a normalized score reflecting how frequently a user purchases and how recent those purchases are.  Results are saved to `outputs/customer_value.csv` and clearly labelled as `CLV_proxy`.

## Real‑Time Simulation

`src/realtime_engine.py` provides `process_realtime_event(event_dict)` which:
1. Updates the user’s event history.
2. Re‑calculates behavioral features & synthetic sentiment.
3. Updates the Transformer embedding (single‑step forward pass).
4. Re‑assigns the customer segment.
5. Generates a fresh recommendation.
6. Logs the recommendation in `outputs/recommendation_logs.csv`.
A demo in `main.py` shows how recommendations evolve for a single user as they view, add‑to‑cart and purchase.

## Causal Analysis

We simulate a **randomised A/B test**:
* **Treatment** – user receives the personalized recommendation from stage 3.
* **Control** – user receives a baseline (most‑popular) recommendation.
Outcomes (click, add‑to‑cart, transaction) are recorded in the generated logs.  Using simple difference‑in‑means (or DoWhy if installed) we estimate the **Average Treatment Effect (ATE)** and report uplift, confidence intervals, and sample sizes in `outputs/causal_results.csv` and `outputs/causal_effect.png`.

---

## Output Files

All generated artefacts are placed under `outputs/`:
* `processed_events.csv`
* `user_features.csv`
* `user_sequences.csv`
* `item_statistics.csv`
* `synthetic_sentiment.csv`
* `customer_value.csv`
* `user_embeddings.csv`
* `customer_segments.csv`
* `segment_statistics.csv`
* `recommendations.csv`
* `recommendation_logs.csv`
* `causal_results.csv`
* Visualisations (`*.png`)

## Limitations & Future Work

* The sentiment and CLV are synthetic proxies – not real monetary or textual sentiment.
* The Transformer is small for CPU feasibility; larger models could improve representation quality.
* Real‑time engine is a simulation; production systems would require streaming infrastructure.
* Causal inference uses a simple randomized design; observational methods could be explored.

---

## Research Integrity & Data Provenance

* **RetailRocket** provides the *real* click‑stream events.
* **Synthetic Sentiment** is explicitly generated as a behavioral proxy.
* **CLV Proxy** is derived from observed transaction frequency and recency.
* **Recommendation Logs** are produced by our own system, not by a live platform.
* **Treatment/Control Assignment** is simulated for prototype evaluation, not a real‑world experiment.
* **Results** are experimental and should not be presented as production performance.

---

*End of README.*
