"""
02_cluster_industries.py

Embeds, clusters, and auto-names B2B "Industry" labels, then maps the
result back onto every row of the dataset.

This is a hardened version of the original auto-clustering script. The
core logic (junk detection, the comma-spacing heuristic that tells real
multi-word categories apart from comma-joined tag lists, UMAP+HDBSCAN
clustering, centroid-based auto-naming, majority-vote mapping for
multi-tag rows) is unchanged because it was already sound — verified
here against a synthetic stand-in dataset built to match the same
shape (junk markers, "A, B" categories, "A,B" tag lists).

WHAT WAS FIXED vs. the original
---------------------------------
1. matplotlib: `plt.cm.get_cmap("tab20", n)` is deprecated/removed in
   recent matplotlib (3.9+) and would eventually break this script.
   Replaced with the modern colormap API.
2. Model-download robustness: the original only caught ImportError
   around the sentence-transformers embedding step, so if the package
   IS installed but the model download fails (no internet, firewall,
   HF Hub hiccup), the whole pipeline would crash instead of falling
   back to TF-IDF. Now catches that case too, with a clear message.
3. New: saves a `cluster_metadata.pkl` file containing, per cluster,
   a normalized centroid vector in the *original* embedding space (not
   just the UMAP-reduced clustering space), plus the cluster names and
   which embedding backend was used. This is what lets
   03_predict_new_industries.py assign a cluster to brand-new industry
   strings later (nearest-centroid by cosine similarity) without
   re-running the embedding+clustering pipeline from scratch.
4. If the TF-IDF fallback embedding is used (no sentence-transformers
   available), the *fitted* vectorizer is saved too, so new text at
   prediction time is transformed consistently rather than re-fit on
   a different vocabulary.
5. Input file path can be overridden from the command line:
   `python 02_cluster_industries.py path/to/file.csv`

Install dependencies first:
    pip install sentence-transformers umap-learn scikit-learn pandas numpy matplotlib
(scikit-learn >= 1.3 ships HDBSCAN built in, so the separate `hdbscan`
package is only needed as a fallback on older sklearn versions.)
"""

import re
import sys
import pickle
import warnings
from collections import Counter

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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────

INPUT_CSV    = sys.argv[1] if len(sys.argv) > 1 else "globalb2bdataset_cleaned.csv"
OUTPUT_CSV   = "b2b_clustered_final.csv"
OUTPUT_PLOT  = "cluster_visualization.png"
METADATA_PKL = "cluster_metadata.pkl"
INDUSTRY_COL = "Industry"

# "all-MiniLM-L6-v2"  -> fast, ~80MB
# "all-mpnet-base-v2" -> slower (~420MB) but noticeably better semantic
#                        separation; recommended default.
EMBEDDING_MODEL = "all-mpnet-base-v2" #768

# "umap_hdbscan" (recommended — finds natural cluster count, marks outliers)
# "kmeans"       (forces a fixed number of clusters, picked via silhouette)
APPROACH = "umap_hdbscan"

K_MIN, K_MAX = 5, 20

UMAP_CLUSTER_COMPONENTS = 15
UMAP_CLUSTER_NEIGHBORS  = 10
UMAP_CLUSTER_MIN_DIST   = 0.0

UMAP_PLOT_NEIGHBORS = 15
UMAP_PLOT_MIN_DIST  = 0.1

HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES      = 2
# "eom"  -> fewer, larger clusters (default)
# "leaf" -> more, smaller/finer clusters; try if you still see oversized
#           catch-all clusters
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

RANDOM_STATE = 42

JUNK = {"no details", "unclassified", "n/a", "none", "", "nan"}

# Real category names that happen to contain a comma always have a space
# after it ("Food, Beverages & Tobacco"). Comma-joined tag lists from a
# different schema never do ("Robotics,Health Diagnostics,Medical
# Device"). This is a cheap, reliable split signal.
_TAG_JOIN_RE = re.compile(r",(?!\s)")


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    sys.stderr.write(
        f"[ERROR] '{INPUT_CSV}' not found. If your data needs cleaning "
        f"first, run 01_clean_raw_data.py to produce it, or pass a path: "
        f"python 02_cluster_industries.py your_file.csv\n"
    )
    sys.exit(1)

print(f"  Rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")

if INDUSTRY_COL not in df.columns:
    sys.stderr.write(f"[ERROR] Expected an '{INDUSTRY_COL}' column. "
                      f"Found: {df.columns.tolist()}\n")
    sys.exit(1)


def split_into_atoms(industry_text):
    if _TAG_JOIN_RE.search(industry_text):
        return [t.strip() for t in industry_text.split(",") if t.strip()]
    return [industry_text]


df["Industry_clean"] = df[INDUSTRY_COL].fillna("").str.strip()
df["_is_junk"] = df["Industry_clean"].str.lower().isin(JUNK)

non_junk_values = df.loc[~df["_is_junk"], "Industry_clean"].unique().tolist()
industry_to_atoms = {v: split_into_atoms(v) for v in non_junk_values}
n_tag_style = sum(1 for v, atoms in industry_to_atoms.items() if len(atoms) > 1)
unique_labels = sorted({atom for atoms in industry_to_atoms.values() for atom in atoms})

print(f"  Junk/unknown rows: {df['_is_junk'].sum()}")
print(f"  Multi-tag (comma-joined) industry values detected: {n_tag_style}")
print(f"  Unique atomic labels to cluster: {len(unique_labels)}")

if len(unique_labels) < 2:
    sys.stderr.write("[ERROR] Fewer than 2 distinct industry labels found — "
                      "nothing to cluster. Check the Industry column.\n")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — EMBED
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Embedding industry labels")
print("=" * 60)

embedding_backend = None     # "sentence_transformers" or "tfidf_char"
tfidf_vectorizer = None      # only set if the TF-IDF fallback was used


def embed_sentence_transformers(texts):
    from sentence_transformers import SentenceTransformer
    print(f"  Loading model '{EMBEDDING_MODEL}' (downloads once)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Encoding industries (L2-normalized)...")
    return model.encode(texts, show_progress_bar=True, batch_size=64,
                         normalize_embeddings=True)


def embed_tfidf_char(texts):
    """Fallback if sentence-transformers isn't installed/usable. Returns
    the dense normalized matrix AND the fitted vectorizer, since the
    vectorizer must be reused (not refit) on any new text later."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           sublinear_tf=True, min_df=1)
    X = vec.fit_transform(texts).toarray()
    return normalize(X), vec


try:
    embeddings = embed_sentence_transformers(unique_labels)
    embedding_backend = "sentence_transformers"
    print(f"  Embedding shape: {embeddings.shape}")
except ImportError:
    print("  sentence-transformers not installed — falling back to TF-IDF")
    embeddings, tfidf_vectorizer = embed_tfidf_char(unique_labels)
    embedding_backend = "tfidf_char"
except Exception as e:
    # Catches model-download / network failures even when the package
    # IS installed, so a flaky connection doesn't crash the whole run.
    print(f"  sentence-transformers model load failed ({e!r}) — "
          f"falling back to TF-IDF")
    embeddings, tfidf_vectorizer = embed_tfidf_char(unique_labels)
    embedding_backend = "tfidf_char"

embeddings = normalize(embeddings)


# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — CLUSTER
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"STEP 3: Clustering (approach='{APPROACH}')")
print("=" * 60)

cluster_labels = None
cluster_space = None

if APPROACH == "umap_hdbscan":
    try:
        import umap
    except ImportError:
        raise ImportError("Run: pip install umap-learn")

    try:
        from sklearn.cluster import HDBSCAN as _HDBSCAN
        _hdbscan_backend = "sklearn"
    except ImportError:
        try:
            import hdbscan as _hdbscan_module
            _HDBSCAN = _hdbscan_module.HDBSCAN
            _hdbscan_backend = "standalone"
        except ImportError:
            raise ImportError(
                "HDBSCAN not available. Either upgrade scikit-learn "
                "(pip install -U scikit-learn, need >=1.3) or install the "
                "standalone package (pip install hdbscan)."
            )
    print(f"  Using HDBSCAN from: {_hdbscan_backend}")

    print(f"  Reducing to {UMAP_CLUSTER_COMPONENTS}D with UMAP (cosine metric)...")
    cluster_reducer = umap.UMAP(
        n_components=min(UMAP_CLUSTER_COMPONENTS, max(len(unique_labels) - 2, 2)),
        metric="cosine",
        n_neighbors=min(UMAP_CLUSTER_NEIGHBORS, max(len(unique_labels) - 1, 2)),
        min_dist=UMAP_CLUSTER_MIN_DIST,
        random_state=RANDOM_STATE,
    )
    cluster_space = cluster_reducer.fit_transform(embeddings)
    print(f"  Reduced shape: {cluster_space.shape}")

    clusterer = _HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
    )
    cluster_labels = clusterer.fit_predict(cluster_space)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"  Clusters found: {n_clusters}  |  Noise (unassigned): {n_noise} "
          f"({n_noise/len(cluster_labels)*100:.1f}%)")

    valid_mask = cluster_labels != -1
    if valid_mask.sum() > 1 and n_clusters > 1:
        score = silhouette_score(cluster_space[valid_mask], cluster_labels[valid_mask])
        print(f"  Silhouette (excl. noise): {score:.4f}")

    if n_noise / len(cluster_labels) > 0.5:
        print("  Over half the labels are unassigned (noise). Consider lowering "
              "HDBSCAN_MIN_CLUSTER_SIZE or HDBSCAN_MIN_SAMPLES, or — if you have "
              "very few unique labels like in a quick test run — that's expected.")

elif APPROACH == "kmeans":
    print(f"  Testing k from {K_MIN} to {K_MAX} (cosine-consistent, normalized vectors)...")
    silhouette_scores = {}
    k_hi = min(K_MAX, len(unique_labels) - 1)
    for k in range(K_MIN, k_hi + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30, max_iter=1000)
        preds = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, preds, metric="cosine")
        silhouette_scores[k] = score
        marker = " <- best so far" if score == max(silhouette_scores.values()) else ""
        print(f"    k={k:>2}  silhouette={score:.4f}{marker}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    best_score = silhouette_scores[best_k]
    print(f"\n  Best k={best_k}, silhouette={best_score:.4f}")

    final_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=30, max_iter=1000)
    cluster_labels = final_km.fit_predict(embeddings)
    cluster_space = embeddings

    fig, ax = plt.subplots(figsize=(9, 4))
    ks = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    ax.plot(ks, scores, marker="o", linewidth=2, color="#378ADD")
    ax.axvline(best_k, color="#D85A30", linestyle="--", linewidth=1.5,
               label=f"Best k={best_k} (score={best_score:.3f})")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (cosine)")
    ax.set_title("Silhouette score vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sil_path = OUTPUT_PLOT.replace(".png", "_silhouette.png")
    plt.savefig(sil_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Silhouette curve saved -> {sil_path}")

else:
    raise ValueError(f"Unknown APPROACH: {APPROACH}")

cluster_labels = np.asarray(cluster_labels)


# ─────────────────────────────────────────────────────────────────────────
# STEP 4 — AUTO-NAME CLUSTERS
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Auto-naming clusters")
print("=" * 60)

label_to_cluster = dict(zip(unique_labels, cluster_labels.tolist()))
unique_clusters = sorted(set(cluster_labels.tolist()))

cluster_members = {}
cluster_member_idx = {}
cluster_centroids = {}   # cluster_id -> normalized centroid in ORIGINAL embedding space

for c in unique_clusters:
    idx = [i for i, l in enumerate(cluster_labels) if l == c]
    cluster_member_idx[c] = idx
    cluster_members[c] = [unique_labels[i] for i in idx]

cluster_names = {}
for c, members in cluster_members.items():
    if c == -1:
        cluster_names[c] = "Unclustered (noise)"
        continue
    idx = cluster_member_idx[c]
    centroid = embeddings[idx].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    cluster_centroids[c] = centroid
    sims = embeddings[idx] @ centroid
    order = np.argsort(-sims)
    top_members = [members[i] for i in order[:2]]
    cluster_names[c] = " / ".join(top_members)

for c in unique_clusters:
    members = cluster_members[c]
    print(f"\n  Cluster {c} [{len(members)} industries] -> '{cluster_names[c]}'")
    for m in members[:8]:
        print(f"    - {m}")
    if len(members) > 8:
        print(f"    ... and {len(members) - 8} more")


# ─────────────────────────────────────────────────────────────────────────
# STEP 5 — MAP BACK TO FULL DATAFRAME
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Mapping clusters back to all rows")
print("=" * 60)


def resolve_row_cluster(industry_text):
    atoms = industry_to_atoms.get(industry_text, [industry_text])
    atom_clusters = [label_to_cluster.get(a, -1) for a in atoms]
    non_noise = [c for c in atom_clusters if c != -1]
    if not non_noise:
        return -1, []
    primary = Counter(non_noise).most_common(1)[0][0]
    distinct = sorted(set(non_noise))
    return primary, distinct


resolved = df.loc[~df["_is_junk"], "Industry_clean"].apply(resolve_row_cluster)
df["Cluster_ID"] = -2  # default: junk/no-data rows
df.loc[~df["_is_junk"], "Cluster_ID"] = resolved.apply(lambda x: x[0])
df["Cluster_ID"] = df["Cluster_ID"].astype(int)

df["Other_Clusters_Touched"] = ""
multi_mask = ~df["_is_junk"]
multi_vals = resolved.apply(lambda x: ",".join(str(c) for c in x[1] if c != x[0]))
df.loc[multi_mask, "Other_Clusters_Touched"] = multi_vals

cluster_names[-2] = "No Industry Data Provided"
df["Cluster_Name"] = df["Cluster_ID"].map(cluster_names).fillna("Unclassified / No Data")
df.drop(columns=["Industry_clean", "_is_junk"], inplace=True)

print("\n  Cluster distribution:")
for name, count in df["Cluster_Name"].value_counts().items():
    print(f"    {name:<55} {count:>4}  ({count/len(df)*100:.1f}%)")

print(f"\n  Clustered: {(df['Cluster_ID'] >= 0).sum()}  |  Junk/noise/unknown: {(df['Cluster_ID'] < 0).sum()}")


# ─────────────────────────────────────────────────────────────────────────
# STEP 6 — 2D VISUALIZATION (UMAP)
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Generating 2D cluster plot (UMAP projection)")
print("=" * 60)

import umap as umap_for_plot

plot_reducer = umap_for_plot.UMAP(
    n_components=2,
    metric="cosine",
    n_neighbors=min(UMAP_PLOT_NEIGHBORS, max(len(unique_labels) - 1, 2)),
    min_dist=UMAP_PLOT_MIN_DIST,
    random_state=RANDOM_STATE,
)
coords_2d = plot_reducer.fit_transform(embeddings)

unique_cluster_ids = sorted(set(cluster_labels.tolist()))

# Modern colormap API (plt.cm.get_cmap(name, n) is removed in newer
# matplotlib). tab20 is a fixed 20-color palette; cycle through it by
# index for any number of clusters.
_tab20 = matplotlib.colormaps["tab20"]
palette = [_tab20(i % _tab20.N) for i in range(len(unique_cluster_ids))]

fig, ax = plt.subplots(figsize=(14, 9))
legend_patches = []

for idx, c in enumerate(unique_cluster_ids):
    mask = cluster_labels == c
    color = "#aaaaaa" if c == -1 else palette[idx]
    name = cluster_names[c]
    short = name if len(name) <= 35 else name[:32] + "..."
    ax.scatter(
        coords_2d[mask, 0], coords_2d[mask, 1],
        c=[color], s=40, alpha=0.8,
        edgecolors="white", linewidths=0.4,
    )
    legend_patches.append(
        mpatches.Patch(color=color, label=f"C{c} ({mask.sum()}): {short}")
    )

ax.set_title("Industry clusters — UMAP 2D projection", fontsize=14, fontweight="bold")
ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")
ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1),
          loc="upper left", fontsize=7, frameon=True, title="Clusters")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved -> {OUTPUT_PLOT}")


# ─────────────────────────────────────────────────────────────────────────
# STEP 7 — SAVE CSV
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: Saving output CSV")
print("=" * 60)

df.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved -> {OUTPUT_CSV}")
print(f"  Columns: {df.columns.tolist()}")


# ─────────────────────────────────────────────────────────────────────────
# STEP 8 — SAVE CLUSTER METADATA (for predicting on brand-new strings later)
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8: Saving cluster metadata")
print("=" * 60)

metadata = {
    "embedding_backend": embedding_backend,   # "sentence_transformers" or "tfidf_char"
    "embedding_model_name": EMBEDDING_MODEL if embedding_backend == "sentence_transformers" else None,
    "tfidf_vectorizer": tfidf_vectorizer,      # only non-None for the tfidf_char backend
    "cluster_centroids": cluster_centroids,    # {cluster_id: unit-norm vector}
    "cluster_names": cluster_names,
}
with open(METADATA_PKL, "wb") as f:
    pickle.dump(metadata, f)
print(f"  Saved -> {METADATA_PKL}  (backend used: {embedding_backend})")


print("\nAll done! Next step: python 03_predict_new_industries.py")
