from transformers import pipeline


def create_simple_llm():
    """Create a simple LLM using DistilGPT-2 model."""
    model_name = "distilgpt2"

    generator = pipeline("text-generation", model=model_name, pad_token_id=50256)

    return generator


generator = create_simple_llm()

prompt = "Do you know about AI?"

# Generate text (using max_new_tokens instead of max_length to avoid truncation warnings)
generated_text = generator(
    prompt, max_new_tokens=50, num_return_sequences=1, truncation=True
)

print(generated_text[0]["generated_text"])
