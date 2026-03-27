# African Trust & Safety LLM Challenge — Red-Team Tool

A web-based tool for the [Zindi African Trust & Safety LLM Challenge](https://zindi.africa/competitions/the-african-trust-safety-llm-challenge). Load African language LLMs locally, craft adversarial prompts, validate safety breaks, and export structured submissions.

![Red-Team Tool](https://img.shields.io/badge/Framework-FastAPI-009688?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## Features

- **Local Model Loading** — Load African language LLMs (Swahili, Hausa, Yoruba, Amharic) directly on your GPU via HuggingFace Transformers
- **Streaming Generation** — Real-time token streaming with SSE
- **Refusal Detection** — Automatic detection of model refusals using multilingual keyword matching
- **Validation Runs** — Run prompts multiple times (up to 10×) to confirm break reproducibility
- **Break Database** — SQLite-backed storage for confirmed safety breaks with full CRUD
- **Taxonomy System** — Attack types, risk categories, and subcategories from the challenge spec
- **AI-Powered Features** (optional, requires OpenAI API key):
  - Prompt optimization
  - Auto-categorization of attacks
  - Contextual note generation
  - Translation between languages
- **Export** — Generate Zindi-format markdown submissions from saved breaks

## Screenshots

<details>
<summary>Prompt Lab</summary>
Dark-themed interface with model selector, prompt editor, streaming response display, and validation controls.
</details>

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (models run on GPU)
- ~4-8 GB VRAM depending on the model

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/josephgitau/The-African-Trust-Safety-LLM-Challenge.git
   cd The-African-Trust-Safety-LLM-Challenge
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python run_webapp.py
   ```
   Opens at [http://127.0.0.1:8000](http://127.0.0.1:8000) when using localhost.

### Colab / public exposure (ngrok)

If you are running in Google Colab, you can expose the app with ngrok:

1. Install dependencies:
   ```bash
   pip install pyngrok
   ```
2. Provide your ngrok auth token (optional but recommended):
   ```bash
   export NGROK_AUTHTOKEN="your_token_here"
   ```
3. Start the app with ngrok forwarding:
   ```bash
   python run_webapp.py --host 0.0.0.0 --port 8000 --ngrok --no-browser
   ```

The script will print a public `ngrok` URL like `https://xxxx.ngrok.io`.

## Usage

1. **Load a model** — Select a model from the top bar dropdown and click "Load". The model downloads from HuggingFace on first use.
2. **Write a prompt** — Type your adversarial prompt in the Prompt Lab (in the target language).
3. **Run** — Click "Run" for streaming generation, or "Validate 3×" to confirm reproducibility.
4. **Save breaks** — When you find a safety failure, click "Save Break" to store it with metadata.
5. **Export** — Go to the Export tab to generate a Zindi-format markdown submission.

### Optional: OpenAI Features

Set an OpenAI API key in Settings to enable AI-powered prompt optimization, auto-categorization, translation, and contextual note generation. The key is stored in memory only and never written to disk.

## Supported Models

| Key | Model | Language | Source |
|-----|-------|----------|--------|
| `swahili` | Pawa-Gemma-Swahili-2B | Swahili | [sartifyllc/Pawa-Gemma-Swahili-2B](https://huggingface.co/sartifyllc/Pawa-Gemma-Swahili-2B) |
| `hausa` | N-ATLaS (Hausa) | Hausa | [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) |
| `yoruba` | N-ATLaS (Yoruba) | Yoruba | [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) |
| `amharic` | Amharic-LLAMA | Amharic | [EthioNLP/Amharic_LLAMA_our_data](https://huggingface.co/EthioNLP/Amharic_LLAMA_our_data) |

## Project Structure

```
├── run_webapp.py          # Entry point — launches the FastAPI server
├── config.py              # Model registry and generation defaults
├── model_utils.py         # HuggingFace model loading and inference
├── requirements.txt       # Python dependencies
├── Data/                  # Challenge taxonomy CSVs
│   ├── attack_types.csv
│   ├── risk_categories.csv
│   └── risk_subcategories.csv
└── webapp/
    ├── app.py             # FastAPI application (API endpoints)
    ├── database.py        # SQLite database layer
    ├── openai_utils.py    # OpenAI API integration (optional features)
    └── static/
        ├── index.html     # Single-page frontend
        ├── style.css      # Dark theme styles
        └── app.js         # Frontend logic
```

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS (no build step)
- **Models:** HuggingFace Transformers + PyTorch
- **Database:** SQLite (auto-created on first run)
- **AI Features:** OpenAI API (optional)

## License

MIT
