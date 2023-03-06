"""
This file allow to train or finetune the Transformer.

To use it, first properly prepaire the train and validation sets in the "datasets" folder.
The train and validation must be saved in two separate files.
The naming convention is <dataset-path>/train_data for the training set and <dataset-path>/val_data for the validation set

Below there are already default values defined for data_dir, out_dir
You can train the model with the default values by simply executing this file without any command line parameter: python train.py

To train the model with custom values of data_dir and out_dir, specify the values in command line as arguments of this file: python train.py --data_dir=<data_dir_path> --out_dir=<out_dir_path>
Here <data_dir_path> and <out_dir_path> are just placeholders of the actual values of data_dir and out_dir.

The model also have some parameters:
vocab_size
d_model
num_heads
expansion
num_layers

You can customize them by specifying then in command line in the same way
"""

import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
from model import Transformer, Tokenizer
import os

ROOT = os.getenv("ROOT")
if ROOT is None:
    raise Exception("Error: environment variable ROOT must be set. Please execute setup.py to solve this problem.")

#----------------------------------------------
# Default training variable values
data_dir = os.path.join(ROOT, "datasets/en_fr") # where the train and validation set files are
dataset_name = os.path.basename(data_dir)
out_dir = os.path.join(ROOT, f"out/{dataset_name}") # where the model weights are going to be saved
# Default model parameters
vocab_size = 8000
d_model = 256
num_heads = 8
expansion = 4
num_layers = 4
# Using custom values if specified in command line
exec(open(os.path.join(ROOT, "arg_parser.py"), "r").read())
#----------------------------------------------
train_set_path = os.path.join(data_dir, "train_data")
val_set_path = os.path.join(data_dir, "val_data")
#----------------------------------------------

if not(os.path.exists(train_set_path)):
    raise Exception(f"Error: {train_set_path} doesn't exist. Please execute prepare.py")

if not(os.path.exists(val_set_path)):
    raise Exception(f"Error: {train_set_path} doesn't exist. Please execute prepare.py")


# Loading train and test sets

train_set = tf.data.Dataset.load(train_set_path)
val_set = tf.data.Dataset.load(val_set_path)

# Defining training scheduler according Attention is all you need paper: https://arxiv.org/pdf/1706.03762.pdf
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps
        }
        return config

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    

# Defining custom loss and accuracy functions
def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    #print(f"In masked_loss, loss shape: {loss.shape}")

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    #print(f"In masked_loss, mask shape: {mask.shape}")

    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)


# Defining the Transformer for training
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
start_token = "[START]"
end_token = "[END]"
context_tokenizer = Tokenizer(os.path.join(ROOT, "encoding/context_tokenizer/english_vocab.txt"), reserved_tokens, start_token, end_token)
target_tokenizer = Tokenizer(os.path.join(ROOT, "encoding/target_tokenizer/french_vocab.txt"), reserved_tokens, start_token, end_token)

transformer = Transformer(
    context_tokenizer=context_tokenizer,
    target_tokenizer=target_tokenizer,
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    expansion=expansion,
    num_layers=num_layers
)

# Defining the learning rate scheduler and the optimizer
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Training
transformer.compile(
    optimizer=optimizer,
    loss=masked_loss,
    metrics=[masked_acc, masked_loss]
)

history = transformer.fit(
    train_set.repeat(), 
    epochs=100,
    steps_per_epoch = 100,
    validation_data=val_set,
    validation_steps = 20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5)])


# Plotting learning curves
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.plot(history.history["loss"], label="train loss")
ax.plot(history.history["val_loss"], label="validation loss")
ax.set_xlabel("Epochs")
ax.legend()
ax.grid()


# Saving learned weights
if not(os.path.exists(out_dir)):
    os.makedirs(out_dir)

transformer.save_weights(f"{out_dir}/weigths")
print(f"Training completed, trained weights have been saved to {out_dir}/weigths")