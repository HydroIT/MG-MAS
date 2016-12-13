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
            # LSTM layer, returns a matching sequence of vectors of dimension hidden_dim
            self.model.add(LSTM(self.hidden_dim, input_shape=(self.timesteps, MidiLSTM.input_dim)))
            # Dropout layer (forget some samples, prevents overfitting)
            self.model.add(Dropout(0.15))
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
        """
        Loads given weight file for the LSTM
        """
        self.model.load_weights(weight_file)

    def train(self, epochs=100, n=20):
        """
        Trains the LSTM network on #n samples (chosen randomly) of the instantiated composer for given epochs.
        """
        x_raw, raw_length = MidiLSTM._get_notes_from_composer(self.composer_idx, n)  # Get data
        x_train = list()
        y_train = list()
        for sample in x_raw:  # Refactor training data to sequences of #timesteps
            x, y = MidiLSTM._sample2sequences(sample, self.timesteps)
            x_train += x
            y_train += y
        x, y = self._reshape_inputs(x_train, y_train)
        self.model.fit(x, y, nb_epoch=epochs, batch_size=self.batch_size, callbacks=self.callbacks_list)

    def train_single(self, stream, epochs=2):
        """
        Trains the LSTM network on given stream for a given number of epochs
        :param stream: music21 stream object
        :param epochs: number of epochs to train (default 2)
        """
        sample = MidiLSTM.stream2inputs(stream)  # Convert to valid inputs
        x, y = MidiLSTM._sample2sequences(sample, self.timesteps)  # Convert to sequences
        if len(x) > 0 and len(y) > 0:  # Sanity check (need some input after conversion...)
            x, y = self._reshape_inputs(x, y)  # reshape for LSTM inputs
            self.model.fit(x, y, nb_epoch=epochs, verbose=0)  # Fit data

    def generate(self, sequence_length=None, iterations=200):
        """
        Generates a music sequence of given length (if None is given (default), creates a sequence of
        length 2*timestamps. Number of iterations is how far to roll the information inside the LSTM
        :return: List of notes (midi values)
        """
        if sequence_length is None:  # Use #timesteps if no sequence length is given
            sequence_length = 2 * self.timesteps
        if iterations - sequence_length < sequence_length:
            iterations += sequence_length  # Ensure we don't include the random initializations in the output
        # Generate random starting pattern
        pattern = np.random.randint(MidiLSTM.rest, MidiLSTM.last_note, size=self.timesteps).tolist()
        result = pattern  # Save the result here
        for _ in range(iterations):  # Iterate enough times
            x = np.array(pattern, dtype='float32')  # Reshape the data
            x = np.reshape(x, (1, self.timesteps, MidiLSTM.input_dim))  # One sample x timesteps x input_dim
            prediction = np.argmax(self.model.predict(x, verbose=0))  # Most matching midi value
            pattern.append(prediction)  # Append to pattern
            pattern = pattern[1:]  # Truncate pattern to fit timesteps
            result.append(prediction)  # Append to result
            if len(result) > sequence_length:  # Truncate to result if needed
                result = result[1:sequence_length]
        return result

    def _reshape_inputs(self, x, y):
        true_x = np.array(x, dtype='float32')  # Create an array of floats from the data
        true_x = np.reshape(true_x, (len(x), self.timesteps, MidiLSTM.input_dim))  # Reshape to match LSTM input
        true_y = np_utils.to_categorical(y, nb_classes=MidiLSTM.vocabulary_size)  # One-hot encode the tags
        return true_x, true_y

    @staticmethod
    def _sample2sequences(sample, timesteps):
        """
        Converts a given sample to a list of inputs and expected outputs.
        Divides the sample into inputs of size timesteps and the expected output is the following input.
        :returns: A list of inputs and a list of expected outputs
        """
        x = list()
        y = list()
        sample_len = len(sample)
        for i in range(sample_len - timesteps):
            seq_in = sample[i:i + timesteps]
            seq_out = sample[i + timesteps]
            x.append(seq_in)
            y.append(seq_out)
        return x, y

    @staticmethod
    def _to_midi_values(m21obj):
        """
        Converts a music21 object to it's MIDI value + 1 (so 0 is rest).
        Only considers first note a chord.
        """
        if m21obj.isNote:
            return m21obj.pitch.midi + 1  # +1 so that 0 is rest
        elif m21obj.isChord:  # Only consider first note in chord
            if len(m21obj.pitches) > 0:
                return m21obj.pitches[0].midi + 1  # +1 so that 0 is rest
            else:  # Invalid
                return MidiLSTM.EOS
        elif m21obj.isRest:
            return MidiLSTM.rest
        return MidiLSTM.EOS

    @staticmethod
    def stream2inputs(stream):
        """
        Converts a music21 stream to a list of midi values that matches the LSTM input
        """
        inputs = list(stream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
        return [MidiLSTM._to_midi_values(x) for x in inputs]

    @staticmethod
    def _get_notes_from_composer(composer_index, n=20):
        """
        Gets a list of notes from the composer index given.
        Samples n samples randomly and extracts the midi values from that.
        If n is None, uses all the data.
        :param composer_index: Matches the composers dictionary in this class
        :param n: Number of samples
        :return: List of lists of notes, and how many samples there are in total
        """
        if composer_index in MidiLSTM.composers.keys():  # Sanity check
            files = music21.corpus.getComposer(MidiLSTM.composers[composer_index])  # File list
            if n is not None:
                files = random.sample(files, n)  # Sample n samples
            data_raw = list()  # Data will be kept here
            for f in files:
                try:
                    mstream = music21.corpus.parse(f)  # Attempt to parse
                    data_raw.append(MidiLSTM.stream2inputs(mstream))  # Convert to midi list and save
                except:
                    continue  # Skip invalid attempts
            return data_raw, len(data_raw)
        else:
            raise Exception("Composer index not found")

    @staticmethod
    def to_midi_stream(notes, durations=None):
        """
        Converts a list of notes (and optional durations) to a playable midi track.
        If no durations are given, all notes will have 1 quarter length.
        :param notes: List of notes (midi values)
        :param durations: List of durations (quarter lengths), where len(durations) = len(notes)
        :return: Stream for the file
        """
        stream = music21.stream.Stream()
        piano = music21.stream.Part()
        piano.insert(music21.instrument.Piano())
        if durations is None:
            durations = [1.0] * len(notes)
        for m, d in zip(notes, durations):
            note = music21.note.Note(m - 1) if m > 0 else music21.note.Rest()
            note.duration.quarterLength = d
            piano.append(note)
        stream.append(piano)
        # stream.show('midi')
        return stream


# Uncomment to see in action (might have bad results :))
# '''
bach = MidiLSTM(7)
bach.load_weights("weights\weights-composer-7-43-2.4315.hdf5")
bach.train(epochs=100, n=4)
result = bach.generate()
print(result)
MidiLSTM.to_midi_stream(result)
# '''
# EOF
