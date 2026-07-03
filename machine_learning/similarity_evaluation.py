import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, labels=None, title="Embedding Visualization"):
    """
    Reduce high‑dimensional embeddings to 2D with PCA and plot them.
    Saves the figure as `embedding_plot.png`.
    """
    from sklearn.decomposition import PCA
    # Reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            idx = np.where(labels == lbl)
            plt.scatter(reduced[idx, 0], reduced[idx, 1], label=str(lbl))
        plt.legend()
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("embedding_plot.png")
    plt.close()
    print("Embedding plot saved to embedding_plot.png")
    
def evaluate_similarity(text_pairs, category_mapping=None, threshold=0.5):
    """Calculate cosine similarity for text pairs and evaluate vector accuracy.

    Args:
        text_pairs (list of tuple): List of (text_a, text_b) pairs.
        category_mapping (dict, optional): Mapping from a text to its category label.
            If provided, ground truth is derived: 1 if both texts share the same
            category, otherwise 0.
        threshold (float, optional): Threshold to binarize cosine scores.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode all texts at once for efficiency
    texts_a = [pair[0] for pair in text_pairs]
    texts_b = [pair[1] for pair in text_pairs]
    embeddings_a = model.encode(texts_a)
    embeddings_b = model.encode(texts_b)

    # Cosine similarity per pair
    cosine_scores = []
    for vec1, vec2 in zip(embeddings_a, embeddings_b):
        score = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        print("score------>",score)

        cosine_scores.append(score)
    cosine_scores = np.array(cosine_scores)
    print("Calculated Cosine Scores:", cosine_scores)

    # Derive ground truth if a category mapping is supplied
    if category_mapping is not None:
        ground_truth = []
        for a, b in text_pairs:
            cat_a = category_mapping.get(a, a)  # fallback to text itself
            cat_b = category_mapping.get(b, b)
            ground_truth.append(1 if cat_a == cat_b else 0)
        ground_truth = np.array(ground_truth)
    else:
        # If no mapping, assume user has supplied ground truth elsewhere
        raise ValueError("Category mapping is required to generate ground truth labels.")

    # Binary predictions based on threshold
    predictions = (cosine_scores >= threshold).astype(int)

    # Metrics
    vector_accuracy = accuracy_score(ground_truth, predictions)
    auc_score = roc_auc_score(ground_truth, cosine_scores)
# Plot ROC curve
    fpr, tpr, _ = roc_curve(ground_truth, cosine_scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()
    print("\n--- Accuracy Report ---")
    print(f"Binary Accuracy (threshold={threshold}): {vector_accuracy * 100:.2f}%")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print("Ground Truth Labels:", ground_truth)
    print("Predicted Labels:", predictions)

    return {
        "cosine_scores": cosine_scores,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "accuracy": vector_accuracy,
        "auc": auc_score,
    }

if __name__ == "__main__":
    # Example usage with the custom category mapping you described
    text_pairs = [
        ("Hospitality, Hospital & Health Care", "hospitality"),
        ("Financial Asset Investing, Accounting Services", "Financial"),
        ("Software Development", "Computer Programming"),  # similar
        ("Agriculture Farming", "Mobile App Development"),   # unrelated
    ]
    # Mapping from each phrase to a normalized category label
    category_mapping = {
        "Hospitality, Hospital & Health Care": "hospitality",
        "hospitality": "hospitality",
        "Financial Asset Investing, Accounting Services": "financial",
        "Financial": "financial",
        "Software Development": "Software Engineering",
        "Computer Programming": "Software Engineering",
        "Agriculture Farming": "Agriculture",
        "Mobile App Development": "Software Engineering",
    }
    results = evaluate_similarity(text_pairs, category_mapping, threshold=0.7)
    # Visualize embeddings for both sides
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_a = model.encode([pair[0] for pair in text_pairs])
    for pair in text_pairs:
        print("pair-1",pair[1])
        print("pair-2",pair[0])
    emb_b = model.encode([pair[1] for pair in text_pairs])
    import numpy as np
    plot_embeddings(np.vstack([emb_a, emb_b]), 
                    labels=np.array(["A"]*len(emb_a) + ["B"]*len(emb_b)),
                    title="Sentence Embeddings (A vs B)")
