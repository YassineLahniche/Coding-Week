import pytest
import pickle
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "../model/xgb_baseline.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def test_model_loading():
    try:
        model = load_model()
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")

    assert model is not None, "Model failed to load"