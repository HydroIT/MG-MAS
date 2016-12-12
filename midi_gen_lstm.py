import random
import music21
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np

"""
An attempt at using deep LSTM network to learn from midi files and generates music thereafter.
Some general ideas are very similar to that of midi_lstm.py, but are reduced for simplicity.
"""

ELEMENT_LENGTH = 4  # 3 for notes (chord of 3) + duration
EOS = tuple([-1] * ELEMENT_LENGTH)  # End of Song symbol

# Similar to the the one in midi_lstm.py. Takes duration into account.
def midi_values(m21obj):
    result = list()
    if m21obj.isNote:
        # +1 to convert to 1-128 (instead of 0-127), divide by 128 to get values in range 0-1
        result.append((m21obj.pitch.midi + 1) / 128.0)
    elif m21obj.isChord:
        result = [(p.midi + 1) / 128.0 for p in m21obj.pitches]
        if len(result) >= ELEMENT_LENGTH:
            result = result[:ELEMENT_LENGTH]
    elif m21obj.isRest:
        result = [0] * ELEMENT_LENGTH
    else:
        return EOS
    while len(result) < ELEMENT_LENGTH:
        result.append(0)  # Pad with 0s
    result[-1] = m21obj.duration.quarterLength / 4.0  # Last element is duration, dont expect more than 4 quarterLengths
    return tuple(result)

# input setup
hidden_dim = 256  # How many cells are passed in the hidden state, etc
batch_size = 128
timesteps = 64  # Consider this many notes at a time to predict the next note
element_size = ELEMENT_LENGTH  # The dimension of each |x_i|

# Checkpoint (for loading weights after training)
filepath = "weights-duration-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# input creation
def get_notes(file_to_parse, xraw):
    """
    Parses a given file's musical notation, creates the input string and adds to the total list of raw X data.
    """
    chorale_stream = music21.corpus.parse(file_to_parse)
    inputs = list(chorale_stream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
    inputs_midi = [midi_values(x) for x in inputs]
    xraw.append(inputs_midi)

print("Creating data...")
chorale_files = random.sample(music21.corpus.getComposer('bach'), 10)
X_raw = list()
for chorale_file in chorale_files:
    get_notes(chorale_file, X_raw)
chorale_files = random.sample(music21.corpus.getComposer('palestrina'), 10)
for chorale_file in chorale_files:
    get_notes(chorale_file, X_raw)
chorale_files = random.sample(music21.corpus.getComposer('trecento'), 10)
for chorale_file in chorale_files:
    get_notes(chorale_file, X_raw)
n_samples = len(X_raw)  # How many samples
# There are 88 tones on a piano, so 88C0 + 88C1 + 88C2 + 88C3 are the vocabulary options
# So roughly ~110,000 (and that's not considering the duration)
# For feasability purposes, we chose 10,000.
n_vocab = 10000

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
X = np.reshape(X, (n_patterns, timesteps, element_size))  # Reshape to match LSTM input
Y = np.array(Y_train, dtype='float32')  # Create an array of floats from the data
Y = np.reshape(Y, (n_patterns, element_size))  # Reshape to match LSTM output

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
# Compile with MSE loss and ada-delta optimizer
model.compile(loss='mse', optimizer='adadelta')

# Some weights to initialize
filename = 'weights-duration-16-0.6360.hdf5'
#model.load_weights(filename)
# Fit the data (takes a while!) and saves the best weights to the files
model.fit(X, Y, nb_epoch=100, batch_size=batch_size, callbacks=callbacks_list)

# Choose a random starting pattern
start = np.random.randint(0, len(X_train) - 1)
pattern = X_train[start]
# generate notes/rest, iterates for 1000 times so the end result is far from our randomly-selected data.
for i in range(1000):
    x = np.array(pattern, dtype='float32')  # Reshape pattern to match network setup
    x = np.reshape(x, (1, timesteps, element_size))
    prediction = model.predict(x, verbose=0).tolist()  # Predict
    print("Predicted {}".format(prediction))  # Verbose...
    pattern.append(tuple(prediction[0]))  # Add to pattern
    pattern = pattern[1:len(pattern)]  # Trim first note in pattern

# Code to convert the pattern to midi file and play it
#TODO - this does not take into account the updates done (i.e. chords & duration), so it does not work right now.
#small fixes once the network is trained...
midis = pattern
print(midis)
stream = music21.stream.Stream()
piano = music21.stream.Part()
piano.insert(music21.instrument.Piano())
for m in midis:
    # m is a tuple of (note, note, note, duration)
    n1 = int(m[0] * 128 + 1)
    n2 = int(m[1] * 128 + 1)
    n3 = int(m[2] * 128 + 1)
    dur = round(m[3] * 4, 1)
    data = None
    if n1 > 0:
        if n2 > 0:
            data = music21.chord.Chord([n1, n2, n3])
        else:
            data = music21.note.Note(n1)
    else:
        data = music21.note.Rest()
    data.duration.quarterLength = dur
    print(data, data.duration.quarterLength)
    piano.append(data)
stream.append(piano)
stream.show('midi')  # TODO - delete this, just for debugging purposes. Return the actual string.
input("wait while midi loads...")

#EOF
