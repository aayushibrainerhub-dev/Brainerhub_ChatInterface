"""
╔══════════════════════════════════════════════════════════════════════╗
║        Full Auto-Clustering Script — B2B Industry Dataset           ║
║        FIXED VERSION — proper, well-separated clusters              ║
║                                                                      ║
║  What changed vs. the original and why:                             ║
║    1. Embeddings are now L2-normalized before clustering/scoring.   ║
║       Sentence embeddings are meant to be compared with cosine      ║
║       similarity — skipping normalization lets vector *length*      ║
║       distort distances, which weakens the clusters themselves.     ║
║    2. Default clustering is now UMAP + HDBSCAN, not forced KMeans.  ║
║       HDBSCAN finds naturally tight groups and marks genuine        ║
║       outliers as noise instead of cramming every label into the    ║
║       nearest centroid regardless of fit.                           ║
║    3. The 2D plot now uses a UMAP projection, not PCA. Your old     ║
║       plot's axes only captured ~12% combined variance (6.8% +      ║
║       5.5%) — that's why clusters looked sparse/scattered even      ║
║       though they may have been reasonably separated in the real    ║
║       384-dim space. UMAP is built to preserve neighborhood/cluster ║
║       structure in 2D, so the picture now matches the clustering.   ║
║    4. Cluster auto-naming now picks the labels closest to each      ║
║       cluster's centroid (most representative), not just the two    ║
║       shortest strings.                                             ║
║    5. KMeans path kept as an option (APPROACH="kmeans") but fixed   ║
║       to be consistent: normalized vectors + cosine-equivalent      ║
║       silhouette throughout.                                        ║
║                                                                      ║
║  Install dependencies first:                                        ║
║    pip install sentence-transformers umap-learn hdbscan              ║
║                 scikit-learn pandas numpy matplotlib                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change these to point to your files
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV    = "globalb2bdataset_cleaned.csv"   # ← put your CSV path here
OUTPUT_CSV   = "b2b_clustered_final.csv"
OUTPUT_PLOT  = "cluster_visualization.png"
INDUSTRY_COL = "Industry"

# "all-MiniLM-L6-v2"  -> fast, ~80MB, default below
# "all-mpnet-base-v2" -> slower (~420MB, ~3-5x runtime) but noticeably
#                        better semantic separation; worth trying if you
#                        still see lexical false-positives like industries
#                        getting grouped just because they share a word
#                        (e.g. "Computer Storage Equipment" + "Warehousing"
#                        both containing "storage").
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# "umap_hdbscan" (recommended — finds natural cluster count, marks outliers)
# "kmeans"       (forces a fixed number of clusters, picked via silhouette)
APPROACH = "umap_hdbscan"

# KMeans search range (only used if APPROACH == "kmeans")
K_MIN, K_MAX = 5, 20

# UMAP settings used for the CLUSTERING space (higher-dim, preserves structure)
UMAP_CLUSTER_COMPONENTS = 15
UMAP_CLUSTER_NEIGHBORS  = 10
UMAP_CLUSTER_MIN_DIST   = 0.0

# UMAP settings used for the 2D VISUALIZATION (separate from clustering space)
UMAP_PLOT_NEIGHBORS = 15
UMAP_PLOT_MIN_DIST  = 0.1

# HDBSCAN settings
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES      = 2

RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

import re

df = pd.read_csv(INPUT_CSV)
print(f"  Rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")

JUNK = {"no details", "unclassified", "n/a", "none", "", "nan"}

df["Industry_clean"] = df[INDUSTRY_COL].fillna("").str.strip()
df["_is_junk"] = df["Industry_clean"].str.lower().isin(JUNK)

# Some rows are real category names that happen to contain a comma
# ("Food, Beverages & Tobacco") — those always have a space after the
# comma. Others are comma-*joined tag lists* from a different schema
# ("Robotics,Health Diagnostics,Medical Device,...") — those never have a
# space after the comma. We use that as a cheap, reliable split signal so
# tag-lists get clustered as individual atomic tags instead of as one
# unique combinatorial string per row (which is what was fragmenting the
# health/biopharma space into 8+ near-duplicate clusters).
_TAG_JOIN_RE = re.compile(r",(?!\s)")


def split_into_atoms(industry_text):
    if _TAG_JOIN_RE.search(industry_text):
        return [t.strip() for t in industry_text.split(",") if t.strip()]
    return [industry_text]


non_junk_values = df.loc[~df["_is_junk"], "Industry_clean"].unique().tolist()
industry_to_atoms = {v: split_into_atoms(v) for v in non_junk_values}

n_tag_style = sum(1 for v, atoms in industry_to_atoms.items() if len(atoms) > 1)

unique_labels = sorted({atom for atoms in industry_to_atoms.values() for atom in atoms})

print(f"  Total rows: {len(df)}")
print(f"  Junk/unknown rows: {df['_is_junk'].sum()}")
print(f"  Multi-tag (comma-joined) industry values detected: {n_tag_style}")
print(f"  Unique atomic labels to cluster: {len(unique_labels)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EMBED (normalized — this matters for clustering quality, not just viz)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Embedding industry labels")
print("=" * 60)


def embed_sentence_transformers(texts):
    from sentence_transformers import SentenceTransformer
    print(f"  Loading model '{EMBEDDING_MODEL}' (downloads once)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Encoding industries (L2-normalized)...")
    # normalize_embeddings=True -> vectors land on the unit sphere, so
    # euclidean distance between them becomes a direct stand-in for cosine
    # distance. This is the standard setup for clustering sentence embeddings.
    return model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )


def embed_tfidf_char(texts):
    """Fallback if sentence-transformers isn't installed."""
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )
    X = vec.fit_transform(texts).toarray()
    return normalize(X)


try:
    embeddings = embed_sentence_transformers(unique_labels)
    print(f"  Embedding shape: {embeddings.shape}")
except ImportError:
    print("  sentence-transformers not found — falling back to TF-IDF")
    embeddings = embed_tfidf_char(unique_labels)

# Belt-and-suspenders: guarantee unit-norm regardless of which path ran above.
embeddings = normalize(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CLUSTER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"STEP 3: Clustering (approach='{APPROACH}')")
print("=" * 60)

cluster_labels = None
cluster_space = None  # the vectors actually used for clustering (for diagnostics)

if APPROACH == "umap_hdbscan":
    try:
        import umap
    except ImportError:
        raise ImportError("Run: pip install umap-learn")

    # Prefer scikit-learn's built-in HDBSCAN (sklearn >= 1.3) — no extra
    # package or C-compiler needed. Fall back to the standalone `hdbscan`
    # package if the sklearn version is too old to have it.
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
        n_components=UMAP_CLUSTER_COMPONENTS,
        metric="cosine",
        n_neighbors=UMAP_CLUSTER_NEIGHBORS,
        min_dist=UMAP_CLUSTER_MIN_DIST,
        random_state=RANDOM_STATE,
    )
    cluster_space = cluster_reducer.fit_transform(embeddings)
    print(f"  Reduced shape: {cluster_space.shape}")

    clusterer = _HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",          # UMAP output space is already euclidean
        cluster_selection_method="eom",
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

    # If almost everything ends up as noise, the min_cluster_size is too strict
    # for your label count — try lowering HDBSCAN_MIN_CLUSTER_SIZE.
    if n_noise / len(cluster_labels) > 0.5:
        print("  ⚠ Over half the labels are unassigned (noise). Consider lowering "
              "HDBSCAN_MIN_CLUSTER_SIZE or HDBSCAN_MIN_SAMPLES.")

elif APPROACH == "kmeans":
    print(f"  Testing k from {K_MIN} to {K_MAX} (cosine-consistent, normalized vectors)...")
    silhouette_scores = {}
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30, max_iter=1000)
        preds = km.fit_predict(embeddings)
        # metric="cosine" is valid here because embeddings are normalized,
        # so this matches what KMeans is implicitly optimizing for.
        score = silhouette_score(embeddings, preds, metric="cosine")
        silhouette_scores[k] = score
        marker = " ◀ best so far" if score == max(silhouette_scores.values()) else ""
        print(f"    k={k:>2}  silhouette={score:.4f}{marker}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    best_score = silhouette_scores[best_k]
    print(f"\n  ✅ Best k={best_k}, silhouette={best_score:.4f}")

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
    print(f"  Silhouette curve saved → {sil_path}")

else:
    raise ValueError(f"Unknown APPROACH: {APPROACH}")

cluster_labels = np.asarray(cluster_labels)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — AUTO-NAME CLUSTERS (by closest-to-centroid representatives)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Auto-naming clusters")
print("=" * 60)

label_to_cluster = dict(zip(unique_labels, cluster_labels.tolist()))
unique_clusters = sorted(set(cluster_labels.tolist()))

cluster_members = {}
cluster_member_idx = {}
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
    # rank members by closeness (cosine sim, since vectors are unit-norm)
    sims = embeddings[idx] @ centroid
    order = np.argsort(-sims)
    top_members = [members[i] for i in order[:2]]
    cluster_names[c] = " / ".join(top_members)

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

from collections import Counter


def resolve_row_cluster(industry_text):
    """Pick the cluster a row belongs to as the most common cluster among
    its atomic tags (a plain single-category row just has one atom, so
    this reduces to a direct lookup in that normal case). Also returns
    the full set of distinct non-noise clusters touched, for transparency
    on rows that came from a multi-tag source string."""
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

# For rows whose source string was a multi-tag list spanning more than one
# cluster, record every cluster touched (comma-separated IDs) so that
# information isn't silently dropped just because we report one primary
# cluster per row. Single-category rows get a blank value here.
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — 2D VISUALISATION (UMAP, not PCA)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Generating 2D cluster plot (UMAP projection)")
print("=" * 60)

import umap as umap_for_plot  # separate import in case APPROACH=="kmeans" skipped it above

plot_reducer = umap_for_plot.UMAP(
    n_components=2,
    metric="cosine",
    n_neighbors=UMAP_PLOT_NEIGHBORS,
    min_dist=UMAP_PLOT_MIN_DIST,
    random_state=RANDOM_STATE,
)
coords_2d = plot_reducer.fit_transform(embeddings)

unique_cluster_ids = sorted(set(cluster_labels.tolist()))
cmap = plt.cm.get_cmap("tab20", max(len(unique_cluster_ids), 1))

fig, ax = plt.subplots(figsize=(14, 9))
legend_patches = []

for idx, c in enumerate(unique_cluster_ids):
    mask = cluster_labels == c
    color = "#aaaaaa" if c == -1 else cmap(idx)
    name  = cluster_names[c]
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
# Lets you assign a cluster to brand-new industry strings later without
# re-running the full embedding+clustering pipeline. Skips noise (-1) rows.

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

labeled = df[df["Cluster_ID"] >= 0].copy()

if labeled["Cluster_ID"].nunique() > 1 and len(labeled) >= 20:
    X = labeled[INDUSTRY_COL].fillna("")
    y = labeled["Cluster_ID"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=5, class_weight="balanced")),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)

    cv = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"Cross-val accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"Test accuracy:      {pipe.score(X_test, y_test):.3f}")

    with open("industry_classifier.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print("Model saved → industry_classifier.pkl")

    new_industries = ["Medicines and Hospitality", "Artificial Intelligence and Machine Learning", "Steel works"]
    preds = pipe.predict(new_industries)
    for ind, pred in zip(new_industries, preds):
        print(f"  '{ind}'  →  Cluster {pred}: {cluster_names.get(pred, '?')}")
else:
    print("  Skipped classifier training (not enough labeled clusters/rows).")