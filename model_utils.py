"""
Model loading and inference utilities for the African Trust & Safety LLM Challenge.
Supports loading models via standard HuggingFace transformers with optional 4-bit quantization.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY


def load_model(model_key: str, quantize_8bit: bool = False, hf_token: str | None = None):
    """
    Load a model and tokenizer from the registry with CPU offloading for T4 compatibility.

    Args:
        model_key: Key from MODELS dict (e.g., 'swahili')
        quantize_8bit: Use 8-bit quantization via bitsandbytes (default False for full precision with offloading)
        hf_token: HuggingFace API token for gated repos (or use HF_TOKEN env var)

    Returns:
        (model, tokenizer) tuple
    """
    assert torch.cuda.is_available(), "CUDA not available! Check PyTorch installation."

    # Get HF token from parameter, environment, or None
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    model_info = MODELS[model_key]
    repo_id = model_info["repo_id"]
    print(f"Loading model: {model_info['name']} ({repo_id})")
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    if hf_token:
        print(f"Using HuggingFace authentication token")

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, token=hf_token)

    if quantize_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
    else:
        # Full precision with CPU offloading to fit T4 GPU
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "15GB", "cpu": "64GB"},  # Limit GPU to 15GB, allow CPU offloading
            trust_remote_code=True,
            token=hf_token,
        )

    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")
    # Note: With device_map="auto", model may be split across GPU and CPU for low VRAM usage

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"VRAM used: {allocated:.2f} GB")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    use_chat_template: bool = True,
) -> str:
    """
    Generate a response from the model given a prompt.
    """
    if use_chat_template and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            messages = [{"from": "human", "value": prompt}]
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                input_text = prompt
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )

    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def quick_test(model, tokenizer, prompts: list[str] | None = None):
    """Run quick test prompts to verify the model is working."""
    if prompts is None:
        prompts = [
            "Habari, jina lako ni nani?",
            "Eleza nchi ya Kenya kwa ufupi.",
            "What languages do you speak?",
        ]

    print("\n" + "=" * 60)
    print("QUICK MODEL TEST")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt, max_new_tokens=200)
        print(f"Response: {response}")
        print()


if __name__ == "__main__":
    model, tokenizer = load_model("swahili", quantize_8bit=False)
    quick_test(model, tokenizer)
