import pytest
import streamlit.web.bootstrap as bootstrap
import threading

def run_streamlit():
    bootstrap.run('app.py', command_line=[], args=[])

def test_streamlit_startup():
    thread = threading.Thread(target=run_streamlit, daemon=True)
    thread.start()
    assert thread.is_alive(), "Streamlit app failed to start"
