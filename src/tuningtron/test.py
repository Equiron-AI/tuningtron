from transformers import AutoConfig


config = AutoConfig.from_pretrained("google/gemma-2-9b-it")
print(config.tie_word_embeddings)
