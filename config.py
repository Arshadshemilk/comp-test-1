"""
Configuration for the African Trust & Safety LLM Challenge.
"""

import os

# Environment fix for OpenMP duplicate lib issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Model Registry ---
MODELS = {
    "swahili": {
        "name": "Pawa-Gemma-Swahili-2B",
        "repo_id": "sartifyllc/Pawa-Gemma-Swahili-2B",
        "language": "Swahili",
        "architecture": "gemma2",
        "chat_template": "chatml",
    },
    "hausa": {
        "name": "N-ATLaS (Hausa)",
        "repo_id": "NCAIR1/N-ATLaS",
        "language": "Hausa",
        "architecture": "unknown",
    },
    "yoruba": {
        "name": "N-ATLaS (Yoruba)",
        "repo_id": "NCAIR1/N-ATLaS",
        "language": "Yoruba",
        "architecture": "unknown",
    },
    "amharic": {
        "name": "Amharic-LLAMA",
        "repo_id": "EthioNLP/Amharic_LLAMA_our_data",
        "language": "Amharic",
        "architecture": "llama",
        "peft_adapter": True,
        "base_model": "meta-llama/Llama-2-7b-hf",  # local base for adapter (requires HF token)
    },
}

# --- Generation Defaults ---
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
