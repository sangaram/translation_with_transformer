import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
from model import Transformer, Tokenizer
import os

ROOT = os.getenv("ROOT")
if ROOT is None:
    raise Exception("Error: environment variable ROOT must be set. Please execute setup.py to solve this problem.")


data_dir = os.path.join(ROOT, "datasets/en_fr")
train_set_path = os.path.join(data_dir, "train_data")
test_set_path = os.path.join(data_dir, "val_data")

if not(os.path.exists(train_set_path)):
    raise Exception(f"Error: {train_set_path} doesn't exist. Please execute prepare.py")

if not(os.path.exists(test_set_path)):
    raise Exception(f"Error: {train_set_path} doesn't exist. Please execute prepare.py")


# Loading train and test sets

train_set = tf.data.Dataset.load(train_set_path)
test_set = tf.data.Dataset.load(test_set_path)

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


d_model = 256
vocab_size = 8000
num_heads = 8
expansion = 4
num_layers = 4

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

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

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
    validation_data=test_set,
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
