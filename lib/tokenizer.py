from transformers import AutoTokenizer

try:
    from transformers import LlamaTokenizer
except ImportError:
    LlamaTokenizer = None


def load_tokenizer(model_name, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(model_name, **kwargs)
    except ValueError as exc:
        # Older LLaMA checkpoints may still advertise the pre-rename class name.
        if "Tokenizer class LLaMATokenizer does not exist" not in str(exc) or LlamaTokenizer is None:
            raise

        llama_kwargs = dict(kwargs)
        llama_kwargs.pop("trust_remote_code", None)
        return LlamaTokenizer.from_pretrained(model_name, **llama_kwargs)
