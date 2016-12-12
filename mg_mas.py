import music21
import random
import datetime
# import tensorflow as tf
# import math


#TODO - idea: MidiMarkovChain with sub markov model, each per track\instrument, lengths are super level
class MidiMarkovChain:
    EOL = None  # Universal end of line symbol
    alpha = 0.12  # Coefficient for probability smoothing

    def __init__(self, music21stream, order=1):
        """
        Initializes an order-1 markov chain from given notes.
        :param parts: List of parts with notes to generate a markov chain transition table from.
        """
        """
        Initializes a markov chain from given raw text.
        :param raw_text: Text to generate a markov chain transition table from.
        :param should_sanitize: Whether to sanitize the text or not.
        :param order: Order of markov chain.
        """
        self.order = order
        self.note_dict = dict()
        self.note_probs = dict()
        self.note_cdfs = dict()
        self.duration_dict = dict()
        self.duration_probs = dict()
        self.duration_cdfs = dict()
        self.note_updates = list()
        self.duration_updates = list()  # Efficiently update only relevant keys
        self.easy_learn(music21stream)
        self.calculate_probability()
        self.calculate_cdf()

    def generate(self, length=40, start_note=None, start_duration=None):
        def random_note():
            note = random.choice(["A", "B", "C", "D", "E", "F", "G"])
            # We want a reasonable sound so we don't include uncommon octaves or accidentals
            # and we give more chances to no accidental color
            octave = random.choice(["3", "4", "5", "6"])
            accidental = random.choice(["", "#", "", "-", "", "#", "", "-", ""])
            return note + accidental + octave
        """
        Generates a piece of notes from this markov states transition.
        :param length: Maximal length of notes to generate
        :param start_note: A starting token (optional, will be chosen at randomly otherwise)
        :return: A piece of notes generated somewhat randomly from the markov state transitions.
        """
        if start_note is None:  # No start state, randomly-uniformly select one (usage of set promises uniformity)
            start_note = random.choice(list(set(self.note_dict.keys())))
        elif start_note not in self.note_dict.keys():  # Bad start state given
            raise LookupError("Cannot find start token in state transitions dictionary - \"" + str(start_note) + "\"")
        if start_duration is None:  # No start state, randomly-uniformly select one (usage of set promises uniformity)
            start_duration = random.choice(list(set(self.duration_dict.keys())))
        elif start_duration not in self.duration_dict.keys():  # Bad start state given
            raise LookupError("Cannot find start token in state transitions dictionary - \"" + str(start_duration) + "\"")

        # Generate notes using the created CDFs
        prev_note, prev_duration = start_note, start_duration  # Keep track of previous state
        gen = list(start_note)
        dur = list(start_duration)
        while len(gen) < length and prev_note != MidiMarkovChain.EOL:
            # explore = random.random()
            # if explore > 1.1:  # Random note (note - only one is produced, regardless of markov chain order
            #     if explore > 1.99:  # Random chord
            #         chord = random.choice([2, 3, 4])
            #         note = tuple([random_note() for _ in range(chord)])
            #         print("Randomly generated chord {}".format(note))
            #     else:
            #         note = random_note()
            #         print("Randomly generated note {}".format(note))
            #     # We want a reasonably long note, so anywhere between 1/16 to 1/2 would be okay.
            #     # a duration of 2 means 2 quarter lengths, i.e. 1/2.
            #     duration = random.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2])
            #     gen.append(note)
            #     dur.append(duration)
            # else:
            rnd = random.random()  # Random number from 0 to 1
            cdf_note, cdf_dur = self.note_cdfs[prev_note], self.duration_cdfs[prev_duration]
            cp_note, cp_dur = cdf_note[0][1], cdf_dur[0][1]
            i = 0
            # Go through the cdf_note until the cumulative probability is higher than the random number 'rnd'.
            while cp_note < rnd:
                i += 1
                cp_note = cdf_note[i][1]
            if cdf_note[i][0] == MidiMarkovChain.EOL:
                return gen, dur  # EOL reached
            gen.append(cdf_note[i][0][-1])  # Add only new addition to gen (no overlap)
            if cdf_note[i][0] in self.note_cdfs.keys():
                prev_note = cdf_note[i][0]  # Update previous state
            i = 0
            while cp_dur < rnd:
                i += 1
                cp_dur = cdf_dur[i][1]
            if cdf_dur[i][0] == MidiMarkovChain.EOL:
                return gen, dur  # EOL reached
            dur.append(cdf_dur[i][0][-1])
            if cdf_dur[i][0] in self.duration_cdfs.keys():
                prev_duration = cdf_dur[i][0]
        return gen, dur

    #TODO - fix
    def likelihood(self, notes, penalize_missing_keys=True, missing_factor=0.025):
        """
        A Psuedo-likelihood function for a given notes. May penalize a token if it
        does not exist in this markov state transition table. Optionally, may raise an exception instead.
        :param notes: piece of notes to evaluate
        :param penalize_missing_keys: whether to penalize missing keys or raise an exception (False = raise exception)
        :param missing_factor: By how much to penalize missing keys (multiplication)
        :return: A value estimating how likely the piece of text is.
        """
        score = 1.0
        for note in notes:
            for i in range(len(notes) - self.order):
                cur_state = [(mobj, mobj.duration.quarterLength) for mobj in notes[i:i + self.order]]  # Get current state
                next_state = [(mobj, mobj.duration.quarterLength) for mobj in notes[i + 1:i + self.order + 1]]  # Next state
                if next_state[-1] == MidiMarkovChain.EOL:  # reached EOL?
                    break
                if cur_state not in self.note_dict.keys():
                    if penalize_missing_keys:  # Penalize if needed
                        score *= missing_factor
                    else:  # Exception if needed
                        raise LookupError("Can't find '" + str(cur_state) + "' in Markov State Transition table (order " +
                                          str(self.order) + ")")
                elif next_state not in self.note_dict[cur_state].keys():
                    if penalize_missing_keys:  # Penalize if needed
                        score *= missing_factor
                    else:  # Exception if needed
                        raise LookupError("Can't find '" + str(cur_state) + " -> " + str(next_state) +
                                          "' in Markov State Transition table (order " + str(self.order) + ")")
                else:  # Psuedo-Likelihood
                    score *= self.note_probs[cur_state][next_state]
        return score

    def calculate_probability(self):
        def calc_prob(dictionary, probs, updates):
            diff_values = len(set(dictionary.keys()))
            for key in updates:
                if key not in probs:
                    probs[key] = dict()
                sub_dict = dictionary[key]
                total = sum(sub_dict.values()) + diff_values * MidiMarkovChain.alpha
                for w, c in sub_dict.items():
                    probs[key][w] = (float(c) + MidiMarkovChain.alpha) / total
            # Original:
            # diff_values = len(set(dictionary.keys()))  # Used to smooth the probabilities
            # for word, sub_dict in dictionary.items():  # Calculate probabilities for each state
            #     if word not in probs:
            #         probs[word] = dict()
            #     total = sum(sub_dict.values())  # Sum over all items for this state
            #     total += diff_values * MidiMarkovChain.alpha  # ditto
            #     for w, c in sub_dict.items():
            #         probs[word][w] = (float(c) + MidiMarkovChain.alpha) / total  # Transform count to probability

        self.note_probs.clear()
        calc_prob(self.note_dict, self.note_probs, self.note_updates)
        print("Done calculating notes probabilities...")
        self.duration_probs.clear()
        calc_prob(self.duration_dict, self.duration_probs, self.duration_updates)
        print("Done calculating durations probabilities...")

    def calculate_cdf(self):
        def calc_cdf(dictionary, cdfs):
            for pred, succ_probs in dictionary.items():
                items = succ_probs.items()
                # Sort the list by the second index in each item and reverse it from highest to lowest.
                sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
                cdf = []
                cumulative_sum = 0.0
                for c, prob in sorted_items:
                    cumulative_sum += prob
                    cdf.append([c, cumulative_sum])
                cdf[-1][1] = 1.0  # For possible rounding errors
                cdfs[pred] = cdf
        # Create CDFs
        self.note_cdfs.clear()
        calc_cdf(self.note_probs, self.note_cdfs)
        print("Done calculating notes cdfs...")
        self.duration_cdfs.clear()
        calc_cdf(self.duration_probs, self.duration_cdfs)
        print("Done calculating durations cdfs...")

    def generate_midi(self, length=40, start_note=None, start_duration=None):
        notes, durs = self.generate(length, start_note, start_duration)
        

        # zip the notes and durs to one note object
        # music21 is a bit selfish and doesn't allow to add a note twice to a stream
        merged_notes = list()
        for n, d in zip(notes, durs):
            note = {'note': n, 'dur': d}
            merged_notes.append(note)

        # delivers an random numbers like [0 < n1..n6 < len(notes)]
        structer_borders = sorted(random.sample(range(len(merged_notes)),  6))
        
        # beginning to put some structure there
        notesAndChords = list()    
        start = 0
        for b in structer_borders:
            while random.randint(0, 5) != 1: # maybe some loops
                notesAndChords.extend(self.create_music21_notes(merged_notes, start, b))
            else: # at least once
                notesAndChords.extend(self.create_music21_notes(merged_notes, start, b))
                start = b
        
        print(len(notesAndChords))

        stream = music21.stream.Stream()
        piano = music21.stream.Part()
        piano.insert(music21.instrument.Flute())
        
        for note in notesAndChords:
            piano.append(note)
        stream.append(piano)



        stream = MidiMarkovChain.toStream(notesAndChords)
        MidiMarkovChain.safe_stream(stream)

    def create_music21_notes(self, notes, start=0, end=10):
        print("start: " + str(start) + " - end: " + str(end))
        notesAndChords = list()    
        for i in range(end - start):
            n = notes[i + start]
            notesAndChords.append(MidiMarkovChain.toNote(n["note"], n["dur"]))
        return notesAndChords

    def easy_learn(self, stream):
        """
        Learns a given stream by formatting it as needed.
        """
        data = list(stream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
        self.learn(data)  # Optional choose note/chord
        # TODO - And learn backwords?
        self.learn(reversed(data))

    def learn(self, part, update=False, log=False):
        """
        Learn from list of notes
        :param part:
        :param update:
        :param log:
        :return:
        """
        def update_dict(dictionary, updates, cur_state, next_state):
            if cur_state not in dictionary:
                dictionary[cur_state] = dict()
            if next_state not in dictionary[cur_state]:
                dictionary[cur_state][next_state] = 1
            else:
                dictionary[cur_state][next_state] += 1
            if cur_state not in updates:
                updates.append(cur_state)

        def notes_duration_tuples(notes_list):
            notes = list()
            duration = list()
            for note in notes_list:
                n, d = MidiMarkovChain.toState(note)
                notes.append(n)
                duration.append(d)
            return tuple(notes), tuple(duration)

        part = [x for x in part if x.duration.quarterLength > 0]  # Remove 0-length notes
        cur_note_state, cur_dur_state = None, None
        for i in range(len(part) - self.order):
            cur_note_state, cur_dur_state = notes_duration_tuples(part[i:i + self.order])
            next_note_state, next_dur_state = notes_duration_tuples(part[i + 1:i + self.order + 1])
            update_dict(self.note_dict, self.note_updates, cur_note_state, next_note_state)
            update_dict(self.duration_dict, self.duration_updates, cur_dur_state, next_dur_state)
        if len(part):
            update_dict(self.note_dict, self.note_updates, cur_note_state, MidiMarkovChain.EOL)  # Add EOL after each part
            # We do not add to duration because we want to be able to draw as much as needed from it

        if update:
            if log:
                print("Learned " + ' '.join(part))
            self.calculate_probability()
            self.calculate_cdf()

    def copy(self):
        import copy
        return copy.deepcopy(self)

    @staticmethod
    def toState(chordNote):
        if chordNote.isNote:
            return chordNote.nameWithOctave, chordNote.duration.quarterLength
        elif chordNote.isChord:
            return tuple([note.nameWithOctave for note in chordNote]), chordNote.duration.quarterLength
        elif chordNote.isRest:
            return "Rest", chordNote.duration.quarterLength
        else:
            return MidiMarkovChain.EOL

    @staticmethod
    def toNote(note, duration):
        # TODO Bad programming behaviour here, open for future ideas
        if isinstance(note, tuple):  # Chord!
            n = music21.chord.Chord(note)
        elif note == "Rest":  # Rest!
            n = music21.note.Rest()
        else:  # Note!
            n = music21.note.Note(note)
        n.duration.quarterLength = duration
        return n

    @staticmethod
    def toStream(notesAndChords):
        stream = music21.stream.Stream()
        piano = music21.stream.Part()

        piano.insert(music21.instrument.Flute())


        for note in notesAndChords:
            piano.append(note)
        stream.append(piano)

        return stream


    @staticmethod
    def safe_stream(stream):
        fn = 'export-' + str(datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")) + '.mid'
        stream.write('midi', fp=fn)
        print("midi saved.")


# vocabulary_size = 300  # Should be enough for 88 notes + most common chords
# embedding_size = 64  # Arbitrarily selected
# batch_size = 8
# lstm_size = 32  # state size?
#
# embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
#                                               stddev=1.0 / math.sqrt(embedding_size)))
# nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
# # Placeholders for inputs
# train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
# train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
# state = tf.zeros([batch_size, lstm.state_size])
# probabilities = []
# loss = 0.0
# for current_batch_of_words in words_in_dataset:
#     # The value of state is updated after processing each batch of words.
#     output, state = lstm(current_batch_of_words, state)
#
#     # The LSTM output can be used to make next word predictions
#     logits = tf.matmul(output, softmax_w) + softmax_b
#     probabilities.append(tf.nn.softmax(logits))
#     loss += loss_function(probabilities, target_words)

#filename = "Beethoven5.mid"
# We will keep 2 markov chains, one for pitch one for duration
# mc = MidiMarkovChain(music21.midi.translate.midiFilePathToStream(filename), order=6)
# print("Learned beethoven")
# mc.easy_learn(music21.midi.translate.midiFilePathToStream("chno0902.mid"))
# print("Learned nocturne")
# mc.easy_learn(music21.midi.translate.midiFilePathToStream("Israel.mid"))
# print("Learned Israel Anthem")
bachs = music21.corpus.getBachChorales()  #music21.corpus.getMonteverdiMadrigals() #
mc = MidiMarkovChain(music21.corpus.parse(bachs[0]), order=6)
for i in range(1, int(len(bachs)/32)):  # 100 samples should be enough for now (studying both reverse and normal!)
    print("Learning bach #" + str(i + 1))
    mc.easy_learn(music21.corpus.parse(bachs[i]))
print("Calculating probabilities...")
mc.calculate_probability()
print("Deriving CDF...")
mc.calculate_cdf()
print("Updated probs and cdfs")

mc.generate_midi(length=30)


#EOF