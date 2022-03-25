# this is a rnn model for character based generation
import os
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence

# build the model


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def train(datapath: str, modelname, epochs: int, seq_length: int = 100, batch_size: int = 64, rnn_units: int = 1024,):

    # load data
    path_to_data = datapath

    # text data
    text = open(path_to_data, 'rb').read().decode(encoding='utf-8')

    # vocabulary
    vocab = sorted(set(text))

    # char to int
    char2idx = {u: i for i, u in enumerate(vocab)}

    # int to char
    idx2char = np.array(vocab)

    # text to int
    def text_to_int(text):
        return np.array([char2idx[c] for c in text])

    # text as int
    text_as_int = text_to_int(text)

    # int to text
    def int_to_text(ints):
        # convert to numpy array if not already
        try:
            ints = ints.array()
        except:
            pass
        return ''.join(idx2char[ints])

    # split text into sequences
    seq_length = seq_length
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    # use the character dataset to create sequences of the text data with the same length
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    # split into input and target
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    # make the dataset
    dataset = sequences.map(split_input_target)

    # split into input and output
    print("[DEBUG]")
    for x, y in dataset.take(2):
        print("\n\nEXAMPLE\n")
        print("INPUT")
        print(int_to_text(x))
        print("\nOUTPUT")
        print(int_to_text(y))
    print()

    # HYPERPARAMETERS
    BATCH_SIZE = batch_size
    VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
    EMBEDDING_DIM = 256
    RNN_UNITS = rnn_units
    BUFFER_SIZE = 10000  # for shuffling

    # make the data
    data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    print("[DEBUG] Model Summary: ")
    model.summary()
    print()

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = f'./src/data/models/{modelname}/'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(data, epochs=epochs, callbacks=[checkpoint_callback])

    return history, vocab, char2idx, idx2char
