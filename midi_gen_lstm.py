import random
import music21
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np

"""
An attempt at using deep LSTM network to learn from midi files and generates music thereafter.
Some general ideas are very similar to that of midi_lstm.py, but are reduced for simplicity.
"""

EOS = -1  # End of Song symbol

# Similar to the the one in midi_lstm.py, but ignores chords for now (each input at every step is of 1 dim - the
# midi value associated with the note; rest is represented as 0)
def midi_values(m21obj):
    if m21obj.isNote:
        return m21obj.pitch.midi
    elif m21obj.isChord:
        # todo - find a better way to represent this
        # for now, just take the first pitch
        return m21obj.pitches[0].midi
    elif m21obj.isRest:
        return 0
    return EOS

# input setup
hidden_dim = 256  # How many cells are passed in the hidden state, etc
batch_size = 128
timesteps = 64  # Consider this many notes at a time to predict the next note

# Checkpoint (for loading weights after training)
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# input creation
def get_notes(file_to_parse, xraw, vocabulary):
    """
    Parses a given file's musical notation, creates the input string and adds to the total list of raw X data.
    Also updated the given vocabulary (so we know how many unique symbols we have)
    """
    chorale_stream = music21.corpus.parse(file_to_parse)
    inputs = list(chorale_stream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
    inputs_midi = [midi_values(x) for x in inputs]
    xraw.append(inputs_midi)
    return vocabulary.union(set(inputs_midi))

print("Creating data...")
chorale_files = random.sample(music21.corpus.getComposer('bach'), 10)
X_raw = list()
vocab = set()
for chorale_file in chorale_files:
    vocab = get_notes(chorale_file, X_raw, vocab)
chorale_files = random.sample(music21.corpus.getComposer('palestrina'), 10)
for chorale_file in chorale_files:
    vocab = get_notes(chorale_file, X_raw, vocab)
chorale_files = random.sample(music21.corpus.getComposer('trecento'), 10)
for chorale_file in chorale_files:
    vocab = get_notes(chorale_file, X_raw, vocab)
n_samples = len(X_raw)  # How many samples
vocab = sorted(vocab)  # Our vocabulary
# Originally used vocab[-1] + 1, but after training for a bit, the weights have a dimension of
# 82, so I've hard-coded this now. If we use word-embeddings, this could look different.
n_vocab = 82  # vocab[-1] + 1

print("Total samples: {}\nTotal vocabulary: {}".format(n_samples, n_vocab))

X_train = list()
Y_train = list()
# Create the actual data. For every #timesteps of notes, the "tag" is the following note.
# Thus the network learns to predict the next note after #timesteps have been given.
# This also splits the data into more samples, obviously, as each sample is now of #timesteps dimension
# With a matching tag of dimension 1 (one-hot encoded)
for sample in X_raw:
    sample_len = len(sample)
    for i in range(sample_len - timesteps):
        seq_in = sample[i:i + timesteps]
        seq_out = sample[i + timesteps]
        X_train.append(seq_in)
        Y_train.append(seq_out)
n_patterns = len(X_train)
print("Total redefined samples of fixed length: {}".format(n_patterns))

X = np.array(X_train, dtype='float32')  # Create an array of floats from the data
X = np.reshape(X, (n_patterns, timesteps, 1))  # Reshape to match LSTM input
Y = np_utils.to_categorical(Y_train, nb_classes=n_vocab)  # One-hot encode the tags

# Create the model (this can, again, be extended to include durations and also be loaded from file)
model = Sequential()  # Sequential model
# First LSTM layer, returns a matching sequence of vectors of dimension hidden_dim
model.add(LSTM(hidden_dim, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# First Dropout layer (forget some samples)
model.add(Dropout(0.15))
# Second LSTM layer, returns a single vector of hidden_dim dimension
model.add(LSTM(hidden_dim))
# Second Dropout layer (forget some samples)
model.add(Dropout(0.1))
# Final layer, reduce to tag-one-hot-encoding and softmax over it
model.add(Dense(Y.shape[1], activation='softmax'))
# Compile with categorical cross-entropy loss and ada-delta optimizer
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

# Some weights to initialize
filename = 'weights-improvement-84-1.2196.hdf5'
model.load_weights(filename)
# Fit the data (takes a while!) and saves the best weights to the files
#model.fit(X, Y, nb_epoch=100, batch_size=batch_size, callbacks=callbacks_list)

# Choose a random starting pattern
start = np.random.randint(0, len(X_train) - 1)
pattern = X_train[start]
# generate notes/rest, iterates for 1000 times so the end result is far from our randomly-selected data.
for i in range(1000):
    x = np.array(pattern, dtype='float32')  # Reshape pattern to match network setup
    x = np.reshape(x, (1, timesteps, 1))
    prediction = np.argmax(model.predict(x, verbose=0))  # Predict
    print("Predicted {}".format(prediction))  # Verbose...
    pattern.append(prediction)  # Add to pattern
    pattern = pattern[1:len(pattern)]  # Trim first note in pattern

# Code to convert the pattern to midi file and play it
midis = pattern
print(midis)
stream = music21.stream.Stream()
piano = music21.stream.Part()
piano.insert(music21.instrument.Piano())
for m in midis:
    piano.append(music21.note.Note(m) if m > 0 else music21.note.Rest())
stream.append(piano)
stream.show('midi')  # TODO - delete this, just for debugging purposes. Return the actual string.
input("wait while midi loads...")

#EOF