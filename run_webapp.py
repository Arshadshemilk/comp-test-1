"""Launch the Red-Team web application."""

import argparse
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uvicorn


def open_browser(url: str):
    """Open browser after a short delay to let the server start."""
    import time
    time.sleep(2)
    webbrowser.open(url)


def start_ngrok(port: int) -> str | None:
    """Start an ngrok tunnel and return the public URL (if available)."""
    try:
        from pyngrok import ngrok
    except ImportError:
        return None

    auth_token = os.environ.get("NGROK_AUTHTOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)

    tunnel = ngrok.connect(port, "http")
    return tunnel.public_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Red-Team FastAPI web app.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Port to bind (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the local browser")
    parser.add_argument("--ngrok", action="store_true", help="Try starting an ngrok tunnel for public access")
    args = parser.parse_args()

    app_url = f"http://{args.host}:{args.port}"
    local_url = app_url if args.host != "0.0.0.0" else f"http://127.0.0.1:{args.port}"

    print("=" * 50)
    print("  Red-Team Tool — African LLM Challenge")
    print(f"  Listening: {app_url}")
    print(f"  Local: {local_url}")

    ngrok_url = None
    if args.ngrok:
        ngrok_url = start_ngrok(args.port)
        if ngrok_url:
            print(f"  ngrok public URL: {ngrok_url}")
        else:
            print("  ngrok is not available (install pyngrok and set NGROK_AUTHTOKEN)")

    print("=" * 50)

    if not args.no_browser and not os.environ.get("COLAB_GPU"):
        threading.Thread(target=open_browser, args=(local_url,), daemon=True).start()

    uvicorn.run("webapp.app:app", host=args.host, port=args.port, reload=False)
