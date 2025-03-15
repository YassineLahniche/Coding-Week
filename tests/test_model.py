import pytest
import unittest.mock as mock

@mock.patch('streamlit.spinner')  # Example of mocking a streamlit component
def test_model_loading(mock_spinner):
    from app import load_model
    model = load_model()
    assert model is not None, "Model failed to load"
