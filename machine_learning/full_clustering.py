"""
╔══════════════════════════════════════════════════════════════════════╗
║        Full Auto-Clustering Script — B2B Industry Dataset           ║
║        No manual labels. No CLUSTER_MAP.                            ║
║                                                                      ║
║  Install dependencies first:                                         ║
║    pip install sentence-transformers umap-learn hdbscan              ║
║                 scikit-learn pandas numpy matplotlib                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change these to point to your files
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV   = "globalb2bdataset_cleaned.csv"   # ← put your CSV path here
OUTPUT_CSV  = "b2b_clustered_final.csv"
OUTPUT_PLOT = "cluster_visualization.png"
INDUSTRY_COL = "Industry"

# Approach: "embeddings" (best), "umap_hdbscan" (auto-k), "tfidf" (no install)
APPROACH = "embeddings"
# embedding: not only find frequency of word but also find out relation or meaning btw the
# TfIdf: It only finds frequency of words

K_MIN, K_MAX = 5, 20
HDBSCAN_MIN_CLUSTER_SIZE = 5

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"  Rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")

JUNK = {"no details", "unclassified", "n/a", "none", "", "nan"}

df["Industry_clean"] = df[INDUSTRY_COL].fillna("").str.strip()
df["_is_junk"] = df["Industry_clean"].str.lower().isin(JUNK)

unique_labels = (
    df.loc[~df["_is_junk"], "Industry_clean"]
    .unique()
    .tolist()
)

print(f"  Total rows: {len(df)}")
print(f"  Junk/unknown rows: {df['_is_junk'].sum()}")
print(f"  Unique industry labels to cluster: {len(unique_labels)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EMBED / VECTORISE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"STEP 2: Vectorising with approach='{APPROACH}'")
print("=" * 60)


def embed_sentence_transformers(texts):
    from sentence_transformers import SentenceTransformer
    print("  Loading model (downloads once ~80MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Encoding industries...")
    return model.encode(texts, show_progress_bar=True, batch_size=64)

# 384-dimensional vectors and also it is suitable for clsuterig algos do not forcig regressive dimensionality reduction, suitable for memory and also for CPU

# ngram_range refers to unigrams and bigrams
def embed_tfidf_char(texts):
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )
    X = vec.fit_transform(texts).toarray()
    return normalize(X)

# ---------------------------------------------------------------------
# Helper for API – embed texts and return clustering labels & shape
# ---------------------------------------------------------------------
from typing import List, Tuple

def run_clustering(texts: List[str], approach: str = "tfidf") -> Tuple[List[int], Tuple[int, int]]:
    """Embed `texts` and compute cluster labels.

    Parameters
    ----------
    texts : List[str]
        Raw documents to cluster.
    approach : str, default "tfidf"
        "tfidf" – character‑level TF‑IDF (fast, no extra model).
        "embeddings" – sentence‑transformer embeddings; falls back to TF‑IDF
        if the ``sentence‑transformers`` package is unavailable.

    Returns
    -------
    labels : List[int]
        Cluster assignment for each document (``-1`` for noise when HDBSCAN).
    embed_shape : Tuple[int, int]
        Shape of the (normalized) embedding matrix.
    """
    # Choose embedding method
    if approach == "embeddings":
        try:
            embeddings = embed_sentence_transformers(texts)
        except Exception:
            embeddings = embed_tfidf_char(texts)
    else:
        embeddings = embed_tfidf_char(texts)

    # Try HDBSCAN first; fallback to KMeans if unavailable
    embeddings = normalize(embeddings)  # Added normalization for better centroid computation
    try:
        # pyrefly: ignore [missing-import]
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        labels = clusterer.fit_predict(embeddings)
    except Exception:
        from sklearn.cluster import KMeans
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

    return labels.tolist(), embeddings.shape


if APPROACH in ("embeddings", "umap_hdbscan"):
    try:
        embeddings = embed_sentence_transformers(unique_labels)
        print(f"  Embedding shape: {embeddings.shape}")
    except ImportError:
        print("  sentence-transformers not found — falling back to TF-IDF")
        embeddings = embed_tfidf_char(unique_labels)
elif APPROACH == "tfidf":
    embeddings = embed_tfidf_char(unique_labels)
    print(f"  TF-IDF shape: {embeddings.shape}")
else:
    raise ValueError(f"Unknown APPROACH: {APPROACH}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CLUSTER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Clustering")
print("=" * 60)

cluster_labels = None


if APPROACH == "embeddings":

    silhouette_scores = {}
    print(f"  Testing k from {K_MIN} to {K_MAX}...")

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=1000)
        preds = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, preds)
        silhouette_scores[k] = score
        marker = " ◀ best so far" if score == max(silhouette_scores.values()) else ""
        print(f"    k={k:>2}  silhouette={score:.4f}{marker}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    best_score = silhouette_scores[best_k]
    print(f"\n  ✅ Best k={best_k}, silhouette={best_score:.4f}")

    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=30, max_iter=1000)
    cluster_labels = final_km.fit_predict(embeddings)

    # Save silhouette curve
    fig, ax = plt.subplots(figsize=(9, 4))
    ks = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    ax.plot(ks, scores, marker="o", linewidth=2, color="#378ADD")
    ax.axvline(best_k, color="#D85A30", linestyle="--", linewidth=1.5,
               label=f"Best k={best_k} (score={best_score:.3f})")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sil_path = OUTPUT_PLOT.replace(".png", "_silhouette.png")
    plt.savefig(sil_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Silhouette curve saved → {sil_path}")
# dpi = dots per inch

elif APPROACH == "umap_hdbscan":
    try:
        import umap
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError("Run: pip install umap-learn hdbscan")

    print("  Reducing with UMAP...")
    reducer = umap.UMAP(
        n_components=10, metric="cosine",
        n_neighbors=15, min_dist=0.0, random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"  Reduced shape: {reduced.shape}")

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=2, metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"  Clusters found: {n_clusters}  |  Noise: {n_noise}")

    valid_mask = cluster_labels != -1
    if valid_mask.sum() > 1:
        score = silhouette_score(reduced[valid_mask], cluster_labels[valid_mask])
        print(f"  Silhouette (excl. noise): {score:.4f}")


elif APPROACH == "tfidf":

    silhouette_scores = {}
    print(f"  Testing k from {K_MIN} to {K_MAX}...")

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=1000)
        preds = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, preds, metric="cosine")
        silhouette_scores[k] = score
        marker = " ◀ best so far" if score == max(silhouette_scores.values()) else ""
        print(f"    k={k:>2}  silhouette={score:.4f}{marker}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    best_score = silhouette_scores[best_k]
    print(f"\n  ✅ Best k={best_k}, silhouette={best_score:.4f}")

    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=15)
    cluster_labels = final_km.fit_predict(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — AUTO-NAME CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Auto-naming clusters")
print("=" * 60)

label_to_cluster = dict(zip(unique_labels, cluster_labels))
unique_clusters = sorted(set(cluster_labels))

cluster_members = {}
for c in unique_clusters:
    members = [unique_labels[i] for i, l in enumerate(cluster_labels) if l == c]
    cluster_members[c] = members

cluster_names = {}
for c, members in cluster_members.items():
    if c == -1:
        cluster_names[c] = "Unclustered (noise)"
        continue
    sorted_members = sorted(members, key=lambda x: (len(x), x))
    cluster_names[c] = " / ".join(sorted_members[:2])

for c in unique_clusters:
    members = cluster_members[c]
    print(f"\n  Cluster {c} [{len(members)} industries] → '{cluster_names[c]}'")
    for m in members[:8]:
        print(f"    • {m}")
    if len(members) > 8:
        print(f"    ... and {len(members) - 8} more")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MAP BACK TO FULL DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Mapping clusters back to all rows")
print("=" * 60)

df["Cluster_ID"]   = df["Industry_clean"].map(label_to_cluster).fillna(-1).astype(int)
df["Cluster_Name"] = df["Cluster_ID"].map(cluster_names).fillna("Unclassified / No Data")
df.drop(columns=["Industry_clean", "_is_junk"], inplace=True)

print("\n  Cluster distribution:")
for name, count in df["Cluster_Name"].value_counts().items():
    print(f"    {name:<55} {count:>4}  ({count/len(df)*100:.1f}%)")

print(f"\n  Clustered: {(df['Cluster_ID'] >= 0).sum()}  |  Junk/unknown: {(df['Cluster_ID'] < 0).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — 2D VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Generating 2D cluster plot")
print("=" * 60)

pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(embeddings)

unique_cluster_ids = sorted(set(cluster_labels))
cmap = plt.cm.get_cmap("tab20", len(unique_cluster_ids))

fig, ax = plt.subplots(figsize=(14, 9))
legend_patches = []

for idx, c in enumerate(unique_cluster_ids):
    mask = cluster_labels == c
    color = "#aaaaaa" if c == -1 else cmap(idx)
    name  = cluster_names[c]
    short = name if len(name) <= 35 else name[:32] + "..."
    ax.scatter(
        coords_2d[mask, 0], coords_2d[mask, 1],
        c=[color], s=40, alpha=0.75,
        edgecolors="white", linewidths=0.4,
    )
    legend_patches.append(
        mpatches.Patch(color=color, label=f"C{c} ({mask.sum()}): {short}")
    )

ax.set_title("Industry clusters — PCA 2D projection", fontsize=14, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1),
          loc="upper left", fontsize=7, frameon=True, title="Clusters")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved → {OUTPUT_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: Saving output CSV")
print("=" * 60)

df.to_csv(OUTPUT_CSV, index=False)
print(f"  ✅ Saved → {OUTPUT_CSV}")
print(f"  Columns: {df.columns.tolist()}")
print("\nAll done!")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — OPTIONAL: TRAIN A CLASSIFIER ON CLUSTER LABELS
# ─────────────────────────────────────────────────────────────────────────────
# Uncomment to train a model that predicts clusters for new industry strings.

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

labeled = df[df["Cluster_ID"] >= 0].copy()
X = labeled[INDUSTRY_COL].fillna("")
y = labeled["Cluster_ID"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=1000, C=5, class_weight="balanced")),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_train, y_train)

cv = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"Cross-val accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
print(f"Test accuracy:      {pipe.score(X_test, y_test):.3f}")

with open("industry_classifier.pkl", "wb") as f:
    pickle.dump(pipe, f)
print("Model saved → industry_classifier.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — VISUALIZE VECTOR CONFIDENCE (ACCURACY SCORE)
# ─────────────────────────────────────────────────────────────────────────────
# Compute prediction confidence for each industry using the trained classifier
X_unique = unique_labels
probs = pipe.predict_proba(X_unique)
# Confidence as maximum probability per sample
confidences = probs.max(axis=1)
# Scale confidence for marker size (e.g., 30 to 120)
size_scaled = 30 + 90 * confidences

# Plot PCA 2D coordinates with marker size reflecting confidence
fig2, ax2 = plt.subplots(figsize=(14, 9))
scatter = ax2.scatter(
    coords_2d[:, 0], coords_2d[:, 1],
    c=["#aaaaaa" if label == -1 else cmap(list(cluster_labels).index(label)) for label in cluster_labels],
    s=size_scaled,
    alpha=0.7,
    edgecolors="black",
    linewidths=0.5,
)
ax2.set_title("Industry vectors with confidence (size) — PCA 2D projection", fontsize=14, fontweight="bold")
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax2.grid(True, alpha=0.2)
plt.tight_layout()
confidence_plot_path = OUTPUT_PLOT.replace('.png', '_confidence.png')
plt.savefig(confidence_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Confidence plot saved → {confidence_plot_path}")

new_industries = ["Medicines and Hospitality", "Artificial Intelligence and Machine Learning", "Steel works"]
preds = pipe.predict(new_industries)
for ind, pred in zip(new_industries, preds):
    print(f"  '{ind}'  →  Cluster {pred}: {cluster_names.get(pred, '?')}")