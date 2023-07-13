# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    # YOUR CODE HERE
    train_data, test_data = imdb['train'], imdb['test']
    train_sentences = []
    train_labels = []
    val_sentences = []
    val_labels = []

    # DO NOT CHANGE THIS CODE
    for s, l in train_data:
        train_sentences.append(s.numpy().decode('utf8'))
        train_labels.append(l.numpy())

    for s, l in test_data:
        val_sentences.append(s.numpy().decode('utf8'))
        val_labels.append(l.numpy())

    # YOUR CODE HERE
    final_train_labels = np.array(train_labels)
    final_val_labels = np.array(val_labels)

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    # Fit your tokenizer with training data
    tokenizer =  Tokenizer(num_words = vocab_size, oov_token=oov_tok) # YOUR CODE HERE

    tokenizer.fit_on_texts(train_sentences)
    tokenizer.word_index
    padded = tokenizer.texts_to_sequences(train_sentences)
    padded = pad_sequences(padded, maxlen=max_length, truncating=trunc_type)
    val_padded = pad_sequences(tokenizer.texts_to_sequences(val_sentences), maxlen=max_length, truncating=trunc_type)


    model = tf.keras.Sequential([
        # YOUR CODE HERE. Do not change the last layer.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=['accuracy'])
    model.summary()

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy') > 0.83 and logs.get('accuracy') > 0.83:
                self.model.stop_training = True
                print("\nReached desired validation accuracy. Stopping training.")

    checkpoint_callback = myCallback()
    model.fit(padded, 
              final_train_labels, 
              epochs=3, 
              batch_size=64,
              validation_data=(val_padded, final_val_labels),
              callbacks = [checkpoint_callback])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A4()
    model.save("model_A4.h5")
