"""Launch the Red-Team web application."""

import os
import sys
import webbrowser
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uvicorn


def open_browser():
    """Open browser after a short delay to let the server start."""
    import time
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    print("=" * 50)
    print("  Red-Team Tool — African LLM Challenge")
    print("  http://127.0.0.1:8000")
    print("=" * 50)
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8000, reload=False)
