import music21
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
import random


class MidiLSTM:
    """
    An attempt at using deep LSTM network to learn from midi files and generates music thereafter.
    This is based on notes alone and is intended for combination with markov chain models for duration.
    """
    EOS = -1  # End of Song symbol
    rest = 0  # Rest symbol
    last_note = 129  # last note midi value (+1 so rest is 0)
    # Composers dictionary (for comparison, generation, etc)
    composers = {0: 'bach', 1: 'beethoven', 2: 'essenFolksong', 3: 'monteverdi',
                 4: 'oneills1850', 5: 'palestrina', 6: 'ryansMammoth', 7: 'trecento'}
    vocabulary_size = 129  # 128 midi values + rest (originally 82 if we want to use previous weights)
    input_dim = 1  # Only one note per time step

    def __init__(self, composer_index, timesteps=64, hidden_dim=256, batch_size=128):
        """
        Initializes a new agent that produces single notes based on LSTM implementation
        and adds duration based on Markov Chain model.
        :param composer_index: Composer index to associate with this agent
        :param timesteps: How many timesteps are in the LSTM model (default 64)
        :param hidden_dim: How big is the hidden dimension for the LSTM model (default 256)
        :param batch_size: How many samples are required for an update (default 128)
        """
        if composer_index in MidiLSTM.composers.keys():
            self.composer_idx = composer_index
            self.hidden_dim = hidden_dim  # Hidden dimensionality between time steps
            self.timesteps = timesteps  # How many time steps are used for prediction
            self.batch_size = batch_size  # How many samples are required for update
            # Create the model (Sequential model)
            self.model = Sequential()
            # First LSTM layer, returns a matching sequence of vectors of dimension hidden_dim
            self.model.add(LSTM(self.hidden_dim, input_shape=(self.timesteps, MidiLSTM.input_dim)))#,
                                #return_sequences=True))
            # First Dropout layer (forget some samples, prevents overfitting)
            self.model.add(Dropout(0.15))
            # Second LSTM layer, returns a matching sequence of vectors of dimension hidden_dim
            #self.model.add(LSTM(self.hidden_dim, return_sequences=True))
            # Second Dropout layer (forget some samples, prevents overfitting)
            #self.model.add(Dropout(0.1))
            # Third LSTM layer, returns one final output vector of hidden_dim (representing the entire sequence)
            #self.model.add(LSTM(self.hidden_dim))
            # Final layer, reduce to tag-one-hot-encoding and softmax over it
            self.model.add(Dense(MidiLSTM.vocabulary_size, activation='softmax'))
            # Compile with categorical cross-entropy loss and ada-delta optimizer
            self.model.compile(loss='categorical_crossentropy', optimizer='adam')
            # Checkpoint (for loading\saving weights after training)
            filepath = "weights-composer-" + str(composer_index) + "-{epoch:02d}-{loss:.4f}.hdf5"
            self.callbacks_list = [ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                                   save_best_only=True, mode='min')]
        else:
            raise Exception("Cannot find specified composer in composer data")

    def load_weights(self, weight_file):
        self.model.load_weights(weight_file)

    def train(self, epochs=100, n=20):
        x_raw, raw_length = MidiLSTM._get_notes_from_composer(self.composer_idx, n)
        x_train = list()
        y_train = list()
        for sample in x_raw:
            sample_len = len(sample)
            for i in range(sample_len - self.timesteps):
                seq_in = sample[i:i + self.timesteps]
                seq_out = sample[i + self.timesteps]
                x_train.append(seq_in)
                y_train.append(seq_out)
        x = np.array(x_train, dtype='float32')  # Create an array of floats from the data
        x = np.reshape(x, (len(x_train), self.timesteps, MidiLSTM.input_dim))  # Reshape to match LSTM input
        y = np_utils.to_categorical(y_train, nb_classes=MidiLSTM.vocabulary_size)  # One-hot encode the tags
        self.model.fit(x, y, nb_epoch=epochs, batch_size=self.batch_size, callbacks=self.callbacks_list)

    def generate(self, sequence_length=None, iterations=1000):
        if sequence_length is None:
            sequence_length = self.timesteps
        pattern = np.random.randint(MidiLSTM.rest, MidiLSTM.last_note, size=self.timesteps).tolist()
        result = pattern
        for _ in range(iterations):
            x = np.array(pattern, dtype='float32')
            x = np.reshape(x, (1, self.timesteps, MidiLSTM.input_dim))  # One sample
            prediction = np.argmax(self.model.predict(x, verbose=0))
            pattern.append(prediction)
            pattern = pattern[1:]
            result.append(prediction)
            if len(result) > sequence_length:
                result = result[1:sequence_length]
        return result

    @staticmethod
    def _to_midi_values(m21obj):
        if m21obj.isNote:
            return m21obj.pitch.midi + 1  # +1 so that 0 is rest
        elif m21obj.isChord:  # Only consider first note in chord
            return m21obj.pitches[0].midi + 1  # +1 so that 0 is rest
        elif m21obj.isRest:
            return MidiLSTM.rest
        return MidiLSTM.EOS

    @staticmethod
    def _get_notes_from_composer(composer_index, n=20):
        if composer_index in MidiLSTM.composers.keys():
            files = music21.corpus.getComposer(MidiLSTM.composers[composer_index])
            if n is not None:
                files = random.sample(files, n)
            data_raw = list()
            for f in files:
                try:
                    mstream = music21.corpus.parse(f)
                    inputs = list(mstream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
                    inputs_midi = [MidiLSTM._to_midi_values(x) for x in inputs]
                    data_raw.append(inputs_midi)
                except:
                    continue
            return data_raw, len(data_raw)
        else:
            raise Exception("Composer index not found")

    @staticmethod
    def _to_midi_stream(sequence):
        stream = music21.stream.Stream()
        piano = music21.stream.Part()
        piano.insert(music21.instrument.Piano())
        for m in sequence:
            piano.append(music21.note.Note(m - 1) if m > 0 else music21.note.Rest())
        stream.append(piano)
        stream.show('midi')
        return stream


# # Train for now...
# bach = MidiLSTM(0)
# bach.load_weights("weights-composer-0-49-2.8121.hdf5")
# bach.train(epochs=50, n=20)
# result = bach.generate()
# print(result)
# MidiLSTM._to_midi_stream(result)

# EOF
