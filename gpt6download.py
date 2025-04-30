from transformers import AutoTokenizer, AutoModelForCausalLM

cache_path = "pre-trained_language_models/gpt-j-6B"

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-j-6B",
    cache_dir=cache_path
)
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    cache_dir=cache_path
)
