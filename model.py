import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text 
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from utils.helpers import download_file
from dataclasses import dataclass
import os
import warnings

warnings.filterwarnings("ignore")

'''
Full Transformer model implementation in this single file.
Reference: https://arxiv.org/pdf/1706.03762.pdf
'''


# Global variables
max_length = 1024
vocab_size = 8000
d_model = 256
num_heads = 8
expansion = 4
num_layers = 4

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, with_mask=False):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.with_mask = with_mask
        self.wQuery = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wKey = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wValue = tf.keras.layers.Dense(d_model, use_bias=False)
        self.mask = None
        if with_mask:
            self.mask = tf.constant(np.triu(np.full((max_length, max_length), float("-inf")), k=1), dtype=tf.float32)


    def call(self, query, key, value):
        
        query = self.wQuery(query)  # (batch_size, seq, d_model)
        key = self.wKey(key)    # (batch_size, seq, d_model)
        value = self.wValue(value)  # (batch_size, seq, d_model)

        if self.mask is None:
            weights = tf.keras.activations.softmax(tf.einsum("ijk,ilk->ijl", query, key)/(self.d_model**0.5))    # (batch_size, seq, seq)
        else:
            L = tf.shape(query)[1]
            weights = tf.keras.activations.softmax(tf.einsum("ijk,ilk->ijl", query, key)/(self.d_model**0.5) + self.mask[:L, :L])    # (batch_size, seq, d_model)

        attention = tf.einsum("ijk,ikl->ijl", weights, value)
        return attention
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, f"d_model must be divisable by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.heads = [
            ScaledDotProductAttention(d_model=d_model//num_heads) for _ in range(num_heads)
        ]

        if num_heads > 1:
            self.out_layer = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value):
        #print(f"query shape: {query.shape}")
        #print(f"key shape: {key.shape}")
        #print(f"value shape: {value.shape}")
        head_attentions = [
            head(query, key, value) for head in self.heads
        ]

        if self.num_heads == 1:
            #print(f"attention shape: {head_attentions[0].shape}")
            return head_attentions[0] # (batch_size, seq, d_model)
        else:
            concat_head = tf.concat(head_attentions, axis=-1)
            #print(f"concat_head shape: {concat_head.shape}")
            attention = self.out_layer(concat_head) # (batch_size, seq, d_model)
            return attention
        
class MaskedMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, f"d_model must be divisable by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.heads = [
            ScaledDotProductAttention(d_model=d_model//num_heads, with_mask=True) for _ in range(num_heads)
        ]

        if num_heads > 1:
            self.out_layer = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value):
        head_attentions = [
            head(query, key, value) for head in self.heads
        ]

        if self.num_heads == 1:
            return head_attentions[0]   # (batch_size, seq, d_model)
        else:
            attention = self.out_layer(tf.concat(head_attentions, axis=-1)) # (batch_size, seq, d_model)
            return attention
        
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(PositionalEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model
        )

        self.encoding = self.positional_encoding(length=1024, depth=d_model)

    def positional_encoding(self, length, depth):
        positions = np.arange(length).reshape(-1, 1)    # (length, 1)
        depths = np.array([2*(i//2) for i in range(depth)]).reshape(1, -1)   # (1, depth)
        angle_rates = 1 / 10000**(depths/depth) # (1, depth)
        angles = positions * angle_rates  # (length, depth)
        encoding = np.cos(angles)
        encoding[:, ::2] = np.sin(encoding[:, ::2])
        return tf.cast(encoding, dtype=tf.float32)

    def call(self, x):
        L = tf.shape(x)[1]
        x = self.embedding(x)
        encoding = self.encoding[:L,:]
        return x + encoding[tf.newaxis,:,:]
    
class EncoderBloc(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=1, expansion=2):
        super(EncoderBloc, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.expansion = expansion
        self.attention_layer = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model*expansion),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x):
        #B, L, D = x.shape
        attention = self.attention_layer(query=x, key=x, value=x)   # (batch_size, seq, d_model)
        #print(f"B={B}, L={L}, D={D}")
        #print(f"x shape: {(B, L, D)}")
        #print(f"attention shape: {attention.shape}")
        attention = self.norm(x + attention)
        out = self.norm(attention + self.ffn(attention))
        return out

class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_tokenizer, vocab_size, d_model, num_heads=1, expansion=2, num_layers=2):
        super(Encoder, self).__init__()
        self.text_tokenizer = text_tokenizer
        self.vocab_size = vocab_size
        self.start_token = text_tokenizer.start_id
        self.end_token = text_tokenizer.end_id
        self.d_model = d_model
        self.num_heads = num_heads
        self.expansion = expansion
        self.embedding = PositionalEmbedding(vocab_size=self.vocab_size, d_model=d_model)
        self.num_layers = num_layers
        self.encoder_blocs = tf.keras.Sequential([
            EncoderBloc(d_model=d_model, num_heads=num_heads, expansion=expansion) for _ in range(num_layers)
        ])

    def call(self, x):
        return self.encoder_blocs(self.embedding(x))

    def convert_input(self, texts):
        batch_size = texts.shape[0]
        texts = tf.convert_to_tensor(texts)
        if len(tf.shape(texts)) == 0:
            texts = texts[tf.newaxis, :]
        context = self.text_tokenizer.tokenize(texts).merge_dims(-2, -1).to_tensor()
        start_tokens = tf.cast(tf.fill([batch_size, 1], self.start_token), dtype=tf.int64)
        end_tokens = tf.cast(tf.fill([batch_size, 1], self.end_token), dtype=tf.int64)
        context = tf.concat([start_tokens, context, end_tokens], axis=-1)
        context = self(context, training=False)
        return context
    
class DecoderBloc(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=1, expansion=2):
        super(DecoderBloc, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.expansion = expansion
        self.masked_attention = MaskedMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model*expansion),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(d_model)
        ])
        
    def call(self, context, x):
        x = self.norm(x + self.masked_attention(query=x, key=x, value=x))
        attention = self.norm(x + self.attention(query=x, key=context, value=context))
        out = self.norm(attention + self.ffn(attention))
        return out

class Decoder(tf.keras.layers.Layer):
    def __init__(self, text_tokenizer, vocab_size, d_model, num_heads=1, expansion=2, num_layers=2):
        super(Decoder, self).__init__()
        self.text_tokenizer = text_tokenizer
        self.vocab_size = vocab_size
        self.start_token = text_tokenizer.start_id
        self.end_token = text_tokenizer.end_id
        self.d_model = d_model
        self.num_heads = num_heads
        self.expansion = expansion
        self.num_layers = num_layers
        self.embedding = PositionalEmbedding(vocab_size=self.vocab_size, d_model=d_model)
        self.decoder_blocs = [
            DecoderBloc(d_model=d_model, num_heads=num_heads, expansion=expansion) for _ in range(num_layers)
        ]
        self.classifier_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, context, x):
        x = self.embedding(x)
        for layer in self.decoder_blocs:
            x = layer(context=context, x=x)
        logits = self.classifier_layer(x)
        return logits

    def get_next_token(self, context, x, done):
        logits = self(context, x, training=False)   # (batch_size, seq, vocab_size)
        tokens = tf.argmax(logits[:, -1, :], axis=-1)[:, tf.newaxis]   # (batch_size, 1)
        done = done | (tokens == self.end_token)
        tokens = tf.where(done, tf.constant(0, dtype=tf.int64), tokens)
        return tokens, done

    def convert_to_text(self, tokens):
        words = self.text_tokenizer.detokenize(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, " ' ", "'")
        result = tf.strings.regex_replace(result, " - ", "-")
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[PAD\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
        return result
    
class Transformer(tf.keras.Model):
    def __init__(self, context_tokenizer, target_tokenizer, vocab_size, d_model, num_heads=1, expansion=2, num_layers=2, **kwargs):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.expansion = expansion
        self.num_layers = num_layers
        self.encoder = Encoder(
            text_tokenizer=context_tokenizer,
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            expansion=expansion,
            num_layers=num_layers
        )
        self.decoder = Decoder(
            text_tokenizer=target_tokenizer,
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            expansion=expansion,
            num_layers=num_layers
        )

        
        if "pretrained" in kwargs and kwargs["pretrained"]:
            url = "http://217.160.46.216/download/tf_english2french"
            model_folder = download_file(url)
            dummy_batch_context = tf.constant(np.random.randint(low=0, high=10, size=(1, 10), dtype=np.int64))
            dummy_batch_targ = tf.constant(np.random.randint(low=0, high=10, size=(1, 10), dtype=np.int64))
            self((dummy_batch_context, dummy_batch_targ)) # To initialize model weights
            self.load_weights(model_folder)

    def call(self, input):
        context, x = input
        context = self.encoder(context)
        logits = self.decoder(context, x)
        return logits

    def translate(self, texts, max_length=max_length):
        context = self.encoder.convert_input(texts) # (batch_size, seq, d_model)
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.decoder.start_token) # (batch_size, 1)
        done = tf.zeros([batch_size, 1], dtype=tf.bool) # (batch_size, 1)
        x = start_tokens
        for _ in range(max_length):
            next_tokens, done = self.decoder.get_next_token(context, x, done)   # (bacth_size, 1)
            if tf.reduce_all(done):
                break
            x = tf.concat([x, next_tokens], axis=-1)
        
        result = self.decoder.convert_to_text(x)
        return result
    
class Tokenizer(tf.Module):
    def __init__(self, vocab_lookup_table, reserved_tokens, start_token, end_token):
        self.bert_tokenizer = tf_text.BertTokenizer(vocab_lookup_table, lower_case=True)
        self.reserved_tokens = reserved_tokens
        self.start_id = tf.argmax(tf.constant(reserved_tokens) == start_token)
        self.end_id = tf.argmax(tf.constant(reserved_tokens) == end_token)

    def tokenize(self, text_input):
        return self.bert_tokenizer.tokenize(text_input)

    def detokenize(self, token_ids):
        return self.bert_tokenizer.detokenize(token_ids)
    
    
@dataclass
class TranslatorConfig():
    d_model:int = d_model
    num_heads:int = num_heads
    num_layers:int = num_layers
    expansion:int = expansion


class Translator(tf.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ROOT = os.getenv("ROOT")
        if ROOT is None:
            raise Exception("Error: environment variable ROOT must be set. Please execute setup.py to solve this problem.")
        
        
        reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
        start_token = "[START]"
        end_token = "[END]"
        context_tokenizer = Tokenizer(os.path.join(ROOT, "encoding/context_tokenizer/english_vocab.txt"), reserved_tokens, start_token, end_token)
        target_tokenizer = Tokenizer(os.path.join(ROOT, "encoding/target_tokenizer/french_vocab.txt"), reserved_tokens, start_token, end_token)

        self.transformer = Transformer(
            context_tokenizer=context_tokenizer,
            target_tokenizer=target_tokenizer,
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            expansion=config.expansion,
            num_layers=config.num_layers,
            pretrained=True
        )

    def __call__(self, texts):
        results = self.transformer.translate(texts)
        return results