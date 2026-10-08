import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# Import project modules
from src import preprocessing, synthetic_sentiment, clv, transformer_embeddings, dynamic_recommendation, customer_segmentation, causal_analysis, realtime_engine

def main():
    # Step 1-4: Preprocess data and generate synthetic features
    print("--- Preprocessing dataset ---")
    preprocessing.preprocess_dataset()
    print("--- Generating synthetic sentiment ---")
    synthetic_sentiment.generate_synthetic_sentiment()
    print("--- Computing CLV proxy ---")
    clv.compute_clv_proxy()

    # Step 5-8: Train transformer and obtain embeddings
    print("--- Training Transformer Embeddings ---")
    sequences_path = Path('outputs') / 'user_sequences.csv'
    if not sequences_path.is_file():
        print("User sequences not found. Exiting.")
        sys.exit(1)
    model, vocab = transformer_embeddings.train_transformer(sequences_path)
    print("--- Generating user embeddings ---")
    transformer_embeddings.generate_user_embeddings(model, vocab, sequences_path)

    # Step 9: Customer segmentation
    print("--- Segmenting customers ---")
    customer_segmentation.segment_customers()

    # Step 10: Dynamic recommendation
    print("--- Generating recommendations ---")
    dynamic_recommendation.generate_recommendations()

    # Step 11: Generate recommendation logs (simulated)
    print("--- Generating recommendation logs ---")
    from src.recommendation_logs import generate_recommendation_logs
    generate_recommendation_logs()

    # Step 12: Causal analysis
    print("--- Performing causal analysis ---")
    causal_analysis.run_causal_analysis()

    # Step 13: Real‑time simulation demo
    print("--- Real‑time engine demonstration ---")
    engine = realtime_engine.RealTimeEngine()
    sample_user = pd.read_csv('outputs/user_features.csv')['visitorid'].iloc[0]
    # Simulate a sequence of events for this user
    demo_events = [
        {"visitorid": sample_user, "event": "view", "itemid": 10, "timestamp": int(time.time()*1000)},
        {"visitorid": sample_user, "event": "addtocart", "itemid": 10, "timestamp": int(time.time()*1000)+1000},
        {"visitorid": sample_user, "event": "transaction", "itemid": 10, "timestamp": int(time.time()*1000)+2000, "transactionid": "tx123"},
    ]
    for ev in demo_events:
        engine.process_realtime_event(ev)
        time.sleep(0.5)  # small pause for readability

    print("\nPipeline completed. All outputs are stored in the 'outputs/' directory.")

if __name__ == "__main__":
    main()
