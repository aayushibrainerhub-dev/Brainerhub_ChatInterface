import pickle
from pathlib import Path
from typing import List

# Path to the trained classifier pickle file (relative to project root)
_MODEL_PATH = Path(__file__).resolve().parents[1] / "machine_learning" / "industry_classifier.pkl"

# Lazy-loaded singleton model
_model = None

def _load_model():
    global _model
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

def predict_industries(industries: List[str]) -> List[int]:
    """Predict cluster IDs for a list of industry strings.

    Parameters
    ----------
    industries: List[str]
        Raw industry descriptions.
    Returns
    -------
    List[int]
        Predicted cluster IDs (matching the IDs used in the clustering pipeline).
    """
    model = _load_model()
    # The saved model is a scikit‑learn Pipeline with a `predict` method.
    return model.predict(industries).tolist()
