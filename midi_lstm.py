import random
import music21
import keras
from keras.utils import np_utils
import numpy as np

"""
An attempt at neural-network-based evaluation function for midi files.
Final result gives a probability for each "genre" from the following list:
0	corpus.getComposer('bach')
1	corpus.getComposer('beethoven')
2	corpus.getComposer('essenFolksong')
3	corpus.getComposer('monteverdi')
4	corpus.getComposer('oneills1850')
5	corpus.getComposer('palestrina')
6	corpus.getComposer('ryansMammoth')
7	corpus.getComposer('trecento')
The probabilities can then be used as evaluating all the different aspects of creativity:
-value: Does this NN consider the midi stream to be > 0.5 for any "genre"?
-novelty: How far is the score given by the NN to the agent's idea of his genre? (i.e. the loss function?)
-surprisingness: Incorporating the agents data, if he hasn't seen anything from genre 4 but got the highest score
there, then quite surprising? etc...

This network does not take into account the length\duration, but can be extended to such quite easily.
"""

# each element in X (before encoding) has this many elements (i.e. |x_i| = 3))
# This allows gathering some chords, etc.
ELEMENT_LENGTH = 3
EOS = tuple([-1] * ELEMENT_LENGTH)  # End of Song symbol, number of elements matches ELEMENT_LENGTH

def midi_values(m21obj):
    """
    Takes a music21 object (note\chord\rest) and returns an ELEMENT_LENGTH-tuple, where
    each element is the midi value for the corresponding note. Chords are broken into the tuple appropriately.
    Rests are denoted by 0, and invalid values are -1.
    This method is quite naive but it is only for preprocessing, so we don't mind :)
    """
    result = list()
    if m21obj.isNote:  # Just a note, add the midi value
        result.append(m21obj.pitch.midi)
    elif m21obj.isChord:
        for p in m21obj.pitches:  # Add all the midi values from the chord, truncating excess values
            if len(result) < ELEMENT_LENGTH:
                result.append(p.midi)
            else:
                break
    elif m21obj.isRest:  # Rest
        result.append(0)
    else:  # Invalid.
        result.append(-1)
    while len(result) < ELEMENT_LENGTH:  # Append more 0's
        result.append(0)
    return tuple(result)


def pad_sequence(seq, maxlen):
    """
    Pads the given sequence (seq) to have exactly maxlen items, and adds the EOS symbol as the last one.
    If there are more than maxlen elements in the sequence, it is truncated
    (only the first maxlen symbol are considered)
    This method is also quite naive, but again - preprocessing....
    """
    if len(seq) >= maxlen:
        seq = seq[:maxlen-1]  # truncate
    seq.append(EOS)  # Append EOS
    t = tuple([0] * ELEMENT_LENGTH)  # Filler
    while len(seq) < maxlen:
        seq.append(t)
    return seq

# input setup
# How many timesteps (or rather, how many notes, etc) are there in a single sequence
# Sequences are padded\truncated to fit to this dimension
timesteps = 768
data_dim = ELEMENT_LENGTH  # The dimension of each timestep
batch_size = 32  # How many sequences to consider at a time (while training)
# input creation


# Shorthand for preprocessing from list of xml music files
# TODO - consider using embedding layers rather than midi values (so we can use mask?)
def get_data(chorale_files, tag, X, Y):
    for chorale_file in chorale_files:
        chorale_stream = music21.corpus.parse(chorale_file)
        inputs = list(chorale_stream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
        int_inputs = pad_sequence([midi_values(x) for x in inputs], timesteps)
        X.append(int_inputs)
        Y.append(tag)
# Training data. X and Y will hold our input sequences and their tags
X = list()
Y = list()
# We sample 50 bach chorales, 50 palestrina and 50 trecento as our training data (just as experiment)
get_data(random.sample(music21.corpus.getBachChorales(), 50), 0, X, Y)
get_data(random.sample(music21.corpus.getComposer('palestrina'), 50), 5, X, Y)
get_data(random.sample(music21.corpus.getComposer('trecento'), 50), 7, X, Y)
random.shuffle(X)  # Shuffle our data
X = np.array(X)  # Convert to numpy array
Y = np_utils.to_categorical(Y, nb_classes=8)  # One-hot encoding for tags

# Test data. 50 bach chorales is enough
X_test = list()
Y_test = list()
get_data(random.sample(music21.corpus.getBachChorales(), 50), 0, X_test, Y_test)
X_test = np.array(X_test)  # to numpy array
Y_test = np_utils.to_categorical(Y_test, nb_classes=8)  # one-hot encoding

# Model creation. Once we're done training this (and the weights) can be loaded with a single command.
model = keras.models.Sequential()  # Sequential model
# First LSTM layer, outputs #timesteps vectors of dimension 64
model.add(keras.layers.LSTM(output_dim=64, activation='tanh', inner_activation='hard_sigmoid',
                            input_shape=(timesteps, data_dim), return_sequences=True))
# Second LSTM layer, takes the encoded vectors from before and outputs a single 32-dimension vector
model.add(keras.layers.LSTM(32))
# Last layer - reduce to 8-dimension vector and softmax over it to get probability
model.add(keras.layers.Dense(8, activation='softmax'))
# Compile with categorical cross-entropy loss and adadelta optimizer
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# Fit the data for 256 epochs
model.fit(X, Y, nb_epoch=256, batch_size=batch_size, verbose=2)

# Some testing printouts
score = model.evaluate(X_test, Y_test)
print(score)  # Score on the test-data
print(model.predict(np.array([X_test[8]]), verbose=1))  # Prediction for single sequence
print(model.predict_classes(np.array([X_test[10]])))  # Prediction-class for single sequence
print(model.predict_proba(np.array([X_test[30]])))  # Prediction probabailities for single sequence

#EOF
