import pytest
from app import load_model

def test_model_loading():
    model = load_model()
    assert model is not None, "Model failed to load"
