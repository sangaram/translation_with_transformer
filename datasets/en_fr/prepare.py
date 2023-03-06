import numpy as np
import tensorflow as tf
from model import Tokenizer

import os

# -------------------------------------------------------
# Some helper functions

def clean_data(data_raw:str):
    contexts, targets = [], []
    for line in data_raw.splitlines():
        context, target = line.split('\t')[:2]
        contexts.append(context)
        targets.append(target)
    
    return np.array(contexts), np.array(targets)

def process_text(
        context,
        target,
        context_tokenizer, 
        target_tokenizer,
        start_token:int=None,
        end_token:int=None
        ):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.cast(tf.fill((batch_size, 1), 2 if start_token is None else start_token), dtype=tf.int64)
    end_tokens = tf.cast(tf.fill((batch_size, 1), 3 if end_token is None else end_token), dtype=tf.int64)
    context = context_tokenizer.tokenize(context).merge_dims(-2, -1)
    context = tf.concat([start_tokens, context, end_tokens], axis=-1).to_tensor()

    target = target_tokenizer.tokenize(target).merge_dims(-2, -1)
    target = tf.concat([start_tokens, target, end_tokens], axis=-1)
    targ_in = target[:,:-1].to_tensor()
    targ_out = target[:,1:].to_tensor()
    return (context, targ_in), targ_out
# -------------------------------------------------------

data_dir = os.path.dirname(__file__)
data_raw = open(os.path.join(data_dir, "en_fr.txt"), 'r').read()

contexts, targets = clean_data(data_raw)

train_size = 0.8
is_train = np.random.uniform(size=(len(targets),)) < 0.8

# Initializing tokenizers
ROOT = os.getenv("ROOT")
if ROOT is None:
    raise Exception("Error: environment variable ROOT must be set. Please execute setup.py to solve this problem.")

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
start_token = "[START]"
end_token = "[END]"
context_tokenizer = Tokenizer(os.path.join(ROOT, "encoding/context_tokenizer/english_vocab.txt"), reserved_tokens, start_token, end_token)
target_tokenizer = Tokenizer(os.path.join(ROOT, "encoding/target_tokenizer/french_vocab.txt"), reserved_tokens, start_token, end_token)

# Building train and validation sets
BUFFER_SIZE = len(contexts)
batch_size = 64
train_set = (
    tf.data.Dataset
    .from_tensor_slices((contexts[is_train], targets[is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(batch_size)
    .map(process_text, tf.data.AUTOTUNE))

val_set = (
    tf.data.Dataset
    .from_tensor_slices((contexts[~is_train], targets[~is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(batch_size)
    .map(process_text, tf.data.AUTOTUNE))

# Saving datasets
train_set.save(os.path.join(data_dir, "train_data"))
val_set.save(os.path.join(data_dir, "val_data"))