'''dummy_predict.py

Utility script that loads the trained industry classifier (pickle file) and evaluates it on a few dummy industry strings.
It prints the predicted cluster ID **and** the corresponding cluster name.

Usage:
    python dummy_predict.py
'''

import os
import sys
import pickle
import pandas as pd

# Paths – should match those used in the clustering script
MODEL_PATH = os.path.join(os.path.dirname(__file__), "industry_classifier.pkl")
CSV_PATH   = os.path.join(os.path.dirname(__file__), "b2b_clustered_final.csv")

def load_model(path: str):
    """Load the pickled sklearn pipeline model."""
    if not os.path.exists(path):
        sys.stderr.write(f"[ERROR] Model file not found at {path}\n")
        sys.exit(1)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def load_cluster_names(csv_path: str) -> dict:
    """Read the saved clustering CSV and build a mapping {cluster_id: cluster_name}."""
    if not os.path.exists(csv_path):
        sys.stderr.write(f"[WARN] CSV file not found at {csv_path}. Cluster names will be unavailable.\n")
        return {}
    df = pd.read_csv(csv_path, dtype={"Cluster_ID": int, "Cluster_Name": str})
    mapping = df.drop_duplicates(subset=["Cluster_ID"]).set_index("Cluster_ID")["Cluster_Name"].to_dict()
    return mapping

def main():
    dummy_industries = [
        "Medicines and Hospitality",
        "Artificial Intelligence and Machine Learning",
        "Steel works",
        "Food, Beverages & Tobacco",
        "Robotics,Health Diagnostics,Medical Device",
    ]

    clf = load_model(MODEL_PATH)
    preds = clf.predict(dummy_industries)

    name_map = load_cluster_names(CSV_PATH)
    for industry, cluster_id in zip(dummy_industries, preds):
        cluster_name = name_map.get(int(cluster_id), "<unknown>")
        print(f"'{industry}' => Cluster ID: {cluster_id} | Name: {cluster_name}")

if __name__ == "__main__":
    main()
