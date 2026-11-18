# Worker functions for the app. They live here (not in app.py) so worker processes can import them.
import joblib
from sliding import score_image

_scaler = None
_model = None


def init(model_path):
    """Runs once in every worker process: load the model."""
    global _scaler, _model
    _scaler, _model = joblib.load(model_path)
    _model.n_jobs = 1          # parallelism comes from many workers, not from inside the forest


def score_chunk(task):
    """task = (square 640x640 image, array of windows). Returns the score of each window."""
    img, wins = task
    return score_image(img, _scaler, _model, wins)
