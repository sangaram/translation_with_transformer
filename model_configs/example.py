"""
A model configuration file template
"""

model_config = {
    'context_vocab_file':"encoding/context_tokenizer/english_vocab.txt",
    'target_vocab_file': "encoding/target_tokenizer/french_vocab.txt",
    'vocab_size': 8000,
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 4,
    'expansion': 4,
    'model_path': "http://217.160.46.216/download/tf_english2french" # Either a path or a url
}