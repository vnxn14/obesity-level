import os
import joblib

CACHE_DIR = "model_cache"

# Bump this whenever data_loader.py, the underlying dataset, or a model's
# training logic changes. It's baked into the cache filename, so bumping it
# invalidates every existing pickle and forces a clean retrain on next run —
# without needing to know the internals of data_loader.py to detect the
# change automatically.
CACHE_VERSION = "v7"


def load_or_train(model_name: str, train_fn):
    """
    Loads a cached (model, scaler, columns, accuracy) tuple from disk if one
    exists for this model + CACHE_VERSION. Otherwise calls train_fn() (a
    zero-arg callable, e.g. train_ann_model) to train fresh, saves the
    result to disk, and returns it.

    model_name: short identifier, e.g. "ann", "knn", "decision_tree"
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{model_name}_{CACHE_VERSION}.pkl")

    if os.path.exists(cache_path):
        try:
            return joblib.load(cache_path)
        except Exception:
            pass  # corrupted or incompatible pickle — fall through and retrain

    result = train_fn()
    joblib.dump(result, cache_path)
    return result