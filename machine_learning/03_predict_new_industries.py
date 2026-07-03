"""
03_predict_new_industries.py

Assigns a cluster to brand-new industry strings (ones that weren't in
the original dataset) without re-running the full embedding+clustering
pipeline.

How it works: loads `cluster_metadata.pkl` (produced by
02_cluster_industries.py), which contains, per cluster, a centroid
vector in the embedding space plus which embedding backend was used.
For each new string, it re-embeds the text with that same backend
(either the actual sentence-transformer model, or the saved/fitted
TF-IDF vectorizer if that was the fallback used during training) and
assigns it to the nearest centroid by cosine similarity. This
generalizes well even to phrasing that wasn't in the training
vocabulary when the sentence-transformer backend was used.

Usage:
    python 03_predict_new_industries.py
    python 03_predict_new_industries.py "Quantum Computing Hardware" "Dairy Farming"
"""

import os
import sys
import pickle
import argparse
import numpy as np

METADATA_PATH = os.path.join(os.path.dirname(__file__), "cluster_metadata.pkl")

DEFAULT_DUMMY_INDUSTRIES = [
    "Medicines and Hospitality",
    "Artificial Intelligence and Machine Learning",
    "Steel works",
    "Food, Beverages & Tobacco",
    "Robotics,Health Diagnostics,Medical Device",
]


def load_metadata(path: str) -> dict:
    if not os.path.exists(path):
        sys.stderr.write(
            f"[ERROR] {path} not found. Run 02_cluster_industries.py first "
            f"to produce it.\n"
        )
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_semantic(industries, metadata):
    """Nearest-centroid prediction in the original embedding space."""
    backend = metadata["embedding_backend"]
    centroids = metadata["cluster_centroids"]          # {cluster_id: unit vector}
    if not centroids:
        sys.stderr.write(
            "[ERROR] cluster_metadata.pkl has no real clusters to compare "
            "against (everything was noise in the original run). Re-run "
            "02_cluster_industries.py with a lower HDBSCAN_MIN_CLUSTER_SIZE "
            "or HDBSCAN_MIN_SAMPLES so at least one cluster forms.\n"
        )
        sys.exit(1)
    cluster_ids = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[c] for c in cluster_ids])  # (n_clusters, dim)

    if backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.stderr.write(
                "[ERROR] cluster_metadata.pkl says the model was trained with "
                "sentence-transformers, but it isn't installed here. Run "
                "`pip install sentence-transformers` and try again.\n"
            )
            sys.exit(1)
        model = SentenceTransformer(metadata["embedding_model_name"])
        vectors = model.encode(industries, normalize_embeddings=True)
    elif backend == "tfidf_char":
        vectorizer = metadata["tfidf_vectorizer"]
        from sklearn.preprocessing import normalize
        vectors = normalize(vectorizer.transform(industries).toarray())
    else:
        raise ValueError(f"Unknown embedding backend in metadata: {backend}")

    # cosine similarity since both sides are unit-normalized
    sims = vectors @ centroid_matrix.T          # (n_new, n_clusters)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    preds = [cluster_ids[i] for i in best_idx]
    return preds, best_sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("industries", nargs="*",
                         help="Industry strings to classify. Defaults to a demo list.")
    args = parser.parse_args()

    industries = args.industries if args.industries else DEFAULT_DUMMY_INDUSTRIES

    metadata = load_metadata(METADATA_PATH)
    name_map = metadata["cluster_names"]

    preds, scores = predict_semantic(industries, metadata)

    for industry, cluster_id, sim in zip(industries, preds, scores):
        cluster_name = name_map.get(int(cluster_id), "<unknown>")
        print(f"'{industry}'\n   -> Cluster {cluster_id}: {cluster_name}  (similarity={sim:.3f})\n")


if __name__ == "__main__":
    main()
