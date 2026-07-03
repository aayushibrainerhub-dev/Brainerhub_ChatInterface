import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Path to the cluster metadata pickle file (relative to project root)
_METADATA_PATH = Path(__file__).resolve().parents[1] / "machine_learning" / "cluster_metadata.pkl"

# Lazy-loaded singleton metadata and models
_metadata = None
_embedding_model = None

def _load_metadata():
    global _metadata
    if _metadata is None:
        with open(_METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)
    return _metadata

def _load_embedding_model(backend: str, model_name: str):
    global _embedding_model
    if _embedding_model is None:
        if backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def predict_cluster_semantic(industries: List[str]) -> List[Dict[str, Any]]:
    """
    Predict cluster assignment for a list of industry strings using the nearest-centroid
    approach in the embedding space (from cluster_metadata.pkl).
    
    Returns a list of dicts:
    [
        {
            "industry": "...",
            "cluster_id": 0,
            "cluster_name": "...",
            "similarity": 0.85
        },
        ...
    ]
    """
    if not industries:
        return []
        
    metadata = _load_metadata()
    backend = metadata["embedding_backend"]
    centroids = metadata["cluster_centroids"]
    print("centroids------", centroids)
    name_map = metadata["cluster_names"]
    print("name_map-------", name_map)
    if not centroids:
        raise ValueError("No cluster centroids found in metadata.")
        
    cluster_ids = list(centroids.keys())
    # Stack centroids to shape (n_clusters, embedding_dim)
    centroid_matrix = np.vstack([centroids[c] for c in cluster_ids])
    print("centroid_matrix-----", centroid_matrix)

    if backend == "sentence_transformers":
        model_name = metadata.get("embedding_model_name", "all-MiniLM-L6-v2")
        model = _load_embedding_model(backend, model_name)
        vectors = model.encode(industries, normalize_embeddings=True)
    elif backend == "tfidf_char":
        vectorizer = metadata["tfidf_vectorizer"]
        from sklearn.preprocessing import normalize
        vectors = normalize(vectorizer.transform(industries).toarray())
    else:
        raise ValueError(f"Unknown embedding backend in metadata: {backend}")
        
    # Since both vectors and centroids are unit-normalized, cosine similarity is just dot product
    sims = np.dot(vectors, centroid_matrix.T)  # shape: (n_industries, n_clusters)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    
    results = []
    for idx, (b_idx, b_sim) in enumerate(zip(best_idx, best_sim)):
        cluster_id = int(cluster_ids[b_idx])
        results.append({
            "industry": industries[idx],
            "cluster_id": cluster_id,
            "cluster_name": name_map.get(cluster_id, "<unknown>"),
            "similarity": float(b_sim)
        })
    return results
