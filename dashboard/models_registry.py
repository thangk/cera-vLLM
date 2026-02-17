"""Recommended models registry with VRAM requirements and metadata."""

RECOMMENDED_MODELS = [
    {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "display_name": "Qwen3 30B-A3B (MoE)",
        "vram_gb": 20,
        "speed": "Fastest",
        "quality": "Good",
        "description": "Mixture-of-Experts model with only 3B active parameters. "
                       "Excellent throughput for batch generation.",
        "requires_hf_token": False,
    },
    {
        "model_id": "Qwen/Qwen3-32B",
        "display_name": "Qwen3 32B (Dense)",
        "vram_gb": 65,
        "speed": "Fast",
        "quality": "Excellent",
        "description": "Dense 32B model with strong reasoning. "
                       "Competitive with GPT-4o on benchmarks.",
        "requires_hf_token": False,
    },
    {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "display_name": "Llama 3.3 70B Instruct",
        "vram_gb": 140,
        "speed": "Moderate",
        "quality": "Excellent",
        "description": "Meta's flagship open model. Natural dialogue and "
                       "strong text generation. Requires HuggingFace token.",
        "requires_hf_token": True,
    },
    {
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "display_name": "Mistral Small 24B Instruct",
        "vram_gb": 48,
        "speed": "Fast",
        "quality": "Good",
        "description": "Efficient model with good balance of quality and speed. "
                       "Apache 2.0 licensed.",
        "requires_hf_token": False,
    },
]
