import random
import music21
import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
from MidiLSTM import MidiLSTM

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

class JudgeLSTM:
    """
    A Judge LSTM network - learns and predicts a genre from the above composers.
    A shallow LSTM network (LSTM -> Dropout -> Fully Connected Layer (softmax))
    """
    ELEMENT_LENGTH = 3  # each element in X (before encoding) has this many elements (i.e. |x_i| = 3))
    EOS = tuple([-1] * ELEMENT_LENGTH)  # End of Song symbol
    rest = 0  # Rest symbol
    n_composers = len(MidiLSTM.composers.keys())  # Total composers to evaluate from

    def __init__(self, timesteps=768, hidden_dim=256, data_dim=ELEMENT_LENGTH, batch_size=16):
        """
        Initializes the LSTM Judge agent.
        :param timesteps: How long is each sequence (sequences shorter than this will be padded to match this length,
        sequences longer than this number will be truncated)
        :param hidden_dim: The hidden dimension for the LSTM layer
        :param data_dim: The data given at each time step
        :param batch_size: How many samples are required to update while training
        """
        self.timesteps = timesteps
        self.data_dim = data_dim
        self.batch_size = batch_size

        # Set up some checkpoint (for training)
        filepath = "weights-classify-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]

        # Model creation (sequential model)
        self.model = keras.models.Sequential()
        # LSTM layer, outputs #timesteps vectors of dimension 128
        self.model.add(keras.layers.LSTM(output_dim=hidden_dim, input_shape=(timesteps, data_dim)))
        # Dropout layer
        self.model.add(keras.layers.Dropout(0.1))
        # Last layer - reduce to 8-dimension vector and softmax over it to get probability
        self.model.add(keras.layers.Dense(JudgeLSTM.n_composers, activation='softmax'))
        # Compile with categorical cross-entropy loss and adadelta optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    def load_weights(self, filename):
        """
        Loads the weights from the HDF5 file to the LSTM model
        """
        self.model.load_weights(filename)

    def train(self, epochs=100, n=20):
        """
        Trains the entire network on creations embedded in music21
        :param epochs: How many epochs to run
        :param n: How many samples from each genre
        """
        x_data = list()
        y_data = list()
        for i in MidiLSTM.composers.keys():
            x, y = self.get_data(i, n)  # Get the data for composer #i
            x_data += x
            y_data += y
        # Fit the model
        self.model.fit(np.array(x_data), np_utils.to_categorical(y_data, nb_classes=JudgeLSTM.n_composers),
                       nb_epoch=epochs, batch_size=self.batch_size, callbacks=self.callbacks_list)

    def fit_single(self, data, label, iterations):
        """
        Fits a single sample (data) and label (mainly for interaction with a multi agent system)
        :param data: A music21 stream, from which the notes and durations will be extracted
        :param label: The matching composer
        :param iterations: How many "epochs" on this specific example (default 2)
        """
        # Convert data to matching types
        x = np.array([JudgeLSTM._stream2ints(data, self.timesteps)])
        y = np_utils.to_categorical([label], nb_classes=JudgeLSTM.n_composers)
        self.model.fit(x, y, nb_epoch=iterations, verbose=0)

    def predict(self, sequence):
        """
        Predicts a probability over the different composers, given the music21 stream
        :param sequence: music21 stream from which notes will be extracted (and padded\truncated to matching length)
        :return: A distribution over the composers (as a list)
        """
        x = np.array([JudgeLSTM._stream2ints(sequence, self.timesteps)])
        return self.model.predict(x).tolist()[0]  # Returns a 2d matrix with 1 row...

    @staticmethod
    def _midi_values(m21obj):
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
                if len(result) < JudgeLSTM.ELEMENT_LENGTH:
                    result.append(p.midi)
                else:
                    break
        elif m21obj.isRest:  # Rest
            result.append(JudgeLSTM.rest)
        else:  # Invalid.
            result = JudgeLSTM.EOS
        while len(result) < JudgeLSTM.ELEMENT_LENGTH:  # Append more "rests"
            result.append(JudgeLSTM.rest)
        return tuple(result)

    @staticmethod
    def _pad_sequence(seq, maxlen):
        """
        Pads the given sequence (seq) to have exactly maxlen items, and adds the EOS symbol as the last one.
        If there are more than maxlen elements in the sequence, it is truncated
        (only the first maxlen symbol are considered)
        This method is also quite naive, but again - preprocessing....
        """
        if len(seq) >= maxlen:
            seq = seq[:maxlen-1]  # truncate
        seq.append(JudgeLSTM.EOS)  # Append EOS
        t = tuple([JudgeLSTM.rest] * JudgeLSTM.ELEMENT_LENGTH)  # Filler
        while len(seq) < maxlen:
            seq.append(t)
        return seq

    @staticmethod
    def _stream2ints(seq, maxlen):
        """
        Converts a given music21 stream to a padded list of ints
        :param seq: music21 stream from which notes and durations will be extracted
        :param maxlen: How long should the output be
        :return: A padded list representing the sequence
        """
        inputs = list(seq.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
        return JudgeLSTM._pad_sequence([JudgeLSTM._midi_values(x) for x in inputs], maxlen)

    def get_data(self, composer_index, n=20):
        """
        Gets n samples from the composer index in padded int-sequence format
        :param composer_index: Matches one of the composers in MidiLSTM.py
        :param n: Number of samples from composer (None for all files, default is 20)
        :return: A tuple (X, Y) for the matching data
        """
        X = list()
        Y = list()
        # Get file list
        files = music21.corpus.getComposer(MidiLSTM.composers[composer_index])
        if n is not None:
            files = random.sample(files, n)  # Sample as required
        for f in files:
            try:
                chorale_stream = music21.corpus.parse(f)  # Parse the file to stream
                X.append(JudgeLSTM._stream2ints(chorale_stream, self.timesteps))  # Convert and pad
                Y.append(composer_index)  # Add matching label
            except:  # Skip errorneous files
                continue
        return X, Y


# Uncomment for a free test :)
# test = JudgeLSTM()
# test.train(epochs=100, n=20)
# test.load_weights("weights-classify-70-0.3340.hdf5")
# print(test.predict(music21.corpus.parse(music21.corpus.getComposer('beethoven')[8])))
# test.fit_single(music21.corpus.parse(music21.corpus.getComposer('bach')[0]), 0, 2)
# print(test.predict(music21.corpus.parse(music21.corpus.getComposer('bach')[0])))

#EOF