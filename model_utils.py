"""
Model loading and inference utilities for the African Trust & Safety LLM Challenge.
Supports loading models via standard HuggingFace transformers with optional 4-bit quantization.
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from config import MODELS, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY

try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


def load_model(model_key: str, quantize_8bit: bool = False, offload_to_cpu: bool = True, hf_token: str | None = None):
    """
    Load a model and tokenizer from the registry using CPU offloading to avoid T4 OOM.

    Args:
        model_key: Key from MODELS dict (e.g., 'swahili')
        quantize_8bit: Use 8-bit quantization via bitsandbytes (default False)
        offload_to_cpu: Offload model layers to CPU when GPU is full (default True)
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

    def _load_base(base_id):
        if quantize_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
            return AutoModelForCausalLM.from_pretrained(
                base_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
            )

        offload_args = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "token": hf_token,
            "max_memory": {0: "12GB", "cpu": "128GB"},
            "offload_folder": os.path.join(os.getcwd(), "offload"),
            "offload_state_dict": True,
            "low_cpu_mem_usage": True,
        }
        if not offload_to_cpu:
            offload_args.pop("offload_folder", None)
            offload_args.pop("offload_state_dict", None)
            offload_args["max_memory"] = {0: "14GB"}

        return AutoModelForCausalLM.from_pretrained(base_id, **offload_args)

    try:
        if model_info.get("peft_adapter", False):
            if not _HAS_PEFT:
                raise ImportError("peft is required for PEFT adapter models. Run: pip install peft")

            base_model_id = model_info.get("base_model")
            if not base_model_id:
                cfg_path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json", token=hf_token)
                with open(cfg_path, "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                base_model_id = adapter_cfg.get("base_model_name_or_path")

            if not base_model_id:
                raise RuntimeError(
                    "Could not determine PEFT adapter base model; set base_model in config.py."
                )

            print(f"Loading PEFT adapter base model: {base_model_id}")
            base_model = _load_base(base_model_id)
            print(f"Applying PEFT adapter weights from {repo_id}")
            model = PeftModel.from_pretrained(
                base_model,
                repo_id,
                device_map="auto",
                torch_dtype=torch.bfloat16 if not quantize_8bit else torch.float16,
                offload_folder=os.path.join(os.getcwd(), "offload"),
                trust_remote_code=True,
                token=hf_token,
            )
        else:
            model = _load_base(repo_id)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print("WARNING: OOM/error during model load:", e)
        if quantize_8bit or model_info.get("peft_adapter", False):
            raise
        print("Retrying with 8-bit quantization to avoid OOM...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            quantization_config=bnb_config,
            device_map="auto",
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
