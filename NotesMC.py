import music21
import random
import pickle


class MidiMarkovChain:
    EOL = None  # Universal end of line symbol
    alpha = 0.12  # Coefficient for probability smoothing

    def __init__(self, music21stream=None, order=1):
        """
        Initializes a markov chain from given MIDI stream
        :param music21stream: stram of music to do a markov chains from
        :param order: Order of markov chain, default 1
        """
        self.order = order
        self.note_dict = dict()
        self.note_probs = dict()
        self.note_cdfs = dict()
        self.duration_dict = dict()
        self.duration_probs = dict()
        self.duration_cdfs = dict()
        self.note_updates = list()
        self.duration_updates = list()
        if music21stream is not None:
            self.easy_learn(music21stream)
            self.update_all()

    def update_all(self):
        """
        Update all probablities and CDF
        """
        self.calculate_probability()
        self.calculate_cdf()
        self.note_updates.clear()
        self.duration_updates.clear()

    def save(self, filename):
        """
        Save all markov chains notes and durations to a file
        :param filename: Name of the file to which the content will be saved
        """
        if len(self.duration_updates) > 0:
            self.update_all()
        with open(filename, "wb") as f:
            pickle.dump(self.note_dict, f)
            pickle.dump(self.note_probs, f)
            pickle.dump(self.note_cdfs, f)
            pickle.dump(self.duration_dict, f)
            pickle.dump(self.duration_probs, f)
            pickle.dump(self.duration_cdfs, f)
            pickle.dump(self.order, f)

    def load(self, filename):
        """
        Load generated Markov chain from a file
        :parem filename: Name of the file to read from
        """
        with open(filename, "rb") as f:
            self.note_dict = pickle.load(f)
            self.note_probs = pickle.load(f)
            self.note_cdfs = pickle.load(f)
            self.duration_dict = pickle.load(f)
            self.duration_probs = pickle.load(f)
            self.duration_cdfs = pickle.load(f)
            self.order = pickle.load(f)

    def generate(self, length=40, start_note=None, start_duration=None):
        """
        Generate list of notes of given length from markov states transition
        :param length: number of notes to generate, default=40
        :param start_note: note to start from, default=None
        :param start_duration: duration of the starting note, default=None
        """
        if start_note is None:  # No start state, randomly-uniformly select one (usage of set promises uniformity)
            start_note = random.choice(list(set(self.note_cdfs.keys())))
        elif start_note not in self.note_dict.keys():  # Bad start state given
            raise LookupError("Cannot find start token in state transitions dictionary - '{}'".format(start_note))
        if start_duration is None:  # No start state, randomly-uniformly select one (usage of set promises uniformity)
            start_duration = random.choice(list(set(self.duration_cdfs.keys())))
        elif start_duration not in self.duration_dict.keys():  # Bad start state given
            raise LookupError("Cannot find start token in state transitions dictionary - '{}'".format(start_duration))

        prev_note, prev_duration = start_note, start_duration
        gen = list(start_note)
        dur = list(start_duration)
        while len(gen) < length and prev_note != MidiMarkovChain.EOL:
            rnd = random.random()
            try:
                cdf_note, cdf_dur = self.note_cdfs[prev_note], self.duration_cdfs[prev_duration]
            except:  # Some error occured! Select randomly again...
                cdf_note = self.note_cdfs[random.choice(list(set(self.note_cdfs.keys())))]
                cdf_dur = self.duration_cdfs[random.choice(list(set(self.duration_cdfs.keys())))]
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

    def likelihood(self, notes, penalize_missing_keys=True, missing_factor=0.99):
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

                csn = []
                for cs in cur_state:
                    if not cs[0].isRest:
                        csn.append(cs[0].nameWithOctave)
                cur_state_notes = tuple(csn)
                nsn = []
                for ns in next_state:
                    if not ns[0].isRest:
                        nsn.append(ns[0].nameWithOctave)
                next_state_notes = tuple(nsn)

                if cur_state_notes not in self.note_dict:
                    if penalize_missing_keys:  # Penalize if needed
                        score *= missing_factor
                    else:  # Exception if needed
                        raise LookupError("Can't find '{}' in Markov State Transition table (order {})".format(
                            cur_state, self.order))
                elif next_state_notes not in self.note_dict[cur_state_notes]:
                    if penalize_missing_keys:
                        score *= missing_factor
                    else:  # Exception if needed
                        raise LookupError("Can't find '{}' -> '{}' in Markov State Transition table (order {})".format(
                            cur_state_notes, next_state_notes))
                else: # Psuedo-Likelihood (Normalized)
                    score *= self.note_probs[cur_state_notes][next_state_notes] /\
                             max(self.note_probs[cur_state_notes].values())
        return score

    def calculate_probability(self):
        """
        Calculates probabilities between notes and durations
        """
        def calc_prob(dictionary, probs, updates):
            diff_values = len(set(dictionary.keys()))
            for key in updates:
                if key not in probs:
                    probs[key] = dict()
                sub_dict = dictionary[key]
                total = sum(sub_dict.values()) + diff_values * MidiMarkovChain.alpha
                for w, c in sub_dict.items():
                    probs[key][w] = (float(c) + MidiMarkovChain.alpha) / total
        calc_prob(self.note_dict, self.note_probs, self.note_updates)
        calc_prob(self.duration_dict, self.duration_probs, self.duration_updates)

    def calculate_cdf(self):
        """
        Calculate cumulative distribution function
        """
        def calc_cdf(dictionary, cdfs, updates):
            for key in updates:
                if key not in cdfs:
                    cdfs[key] = dict()
                items = dictionary[key].items()
                sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
                cdf = []
                cumulative_sum = 0.0
                for c, prob in sorted_items:
                    cumulative_sum += prob
                    cdf.append([c, cumulative_sum])
                cdf[-1][1] = 1.0  # For possible rounding errors
                cdfs[key] = cdf

        calc_cdf(self.note_probs, self.note_cdfs, self.note_updates)
        calc_cdf(self.duration_probs, self.duration_cdfs, self.duration_updates)

    def generate_piece(self, length=40, start_note=None, start_duration=None):
        """
        Generates a set of music21 tones based on the calculated probabilites
        :param length: length of the piece
        :param start_note: optional start note
        :param start_duration: optional start duration
        :param repeats: optional - repeats some parts
        """
        notes, durs = self.generate(length, start_note, start_duration)
        merged_notes = list()
        for n, d in zip(notes, durs):
            note = {'note': n, 'dur': d}
            merged_notes.append(note)
        notesAndChords = self.create_music21_notes(merged_notes, 0, len(merged_notes))
        return notesAndChords

    def create_music21_notes(self, notes, start=0, end=10):
        """
        Converts a given note list to an actual note21 list, which get an unique ID
        :param notes: list of arrays (note, duration)
        :param start: starting point
        :param end: end point
        """
        notesAndChords = list()
        for i in range(end - start):
            n = notes[i + start]
            notesAndChords.append(MidiMarkovChain.toNote(n["note"], n["dur"]))
        return notesAndChords

    def easy_learn(self, stream):
        """
        Learns a given stream by formatting it as needed.
        :param stream: the music stream
        """
        data = list(stream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
        self.learn(data)
        self.learn(reversed(data))

    def learn(self, part, update=False, log=False):
        """
        Learn from list of notes
        :param part:
        :param update: Boolean value signalizing whether to update or not
        :param log: Boolean value signalizing whether to log or not
        :return:
        """
        def update_dict(dictionary, updates, cur_state, next_state):
            # Generic hidden function to update a given dictionary and an updates list with the given cur_state
            # and next_state
            if cur_state not in dictionary:
                dictionary[cur_state] = dict()
            if next_state not in dictionary[cur_state]:
                dictionary[cur_state][next_state] = 1
            else:
                dictionary[cur_state][next_state] += 1
            if cur_state not in updates:
                updates.append(cur_state)

        def notes_duration_tuples(notes_list):
            # Convert a notes list to valid markov chain states
            notes = list()
            duration = list()
            for note in notes_list:
                n, d = MidiMarkovChain.toState(note)
                notes.append(n)
                duration.append(d)
            return tuple(notes), tuple(duration)

        # Remove 0-length notes
        temp = list()
        for x in part:
            try:
                if x.duration.quarterLength > 0:
                    temp.append(x)
            except:
                continue
        part = temp

        cur_note_state, cur_dur_state = None, None
        for i in range(len(part) - self.order - 1):
            cur_note_state, cur_dur_state = notes_duration_tuples(part[i:i + self.order])
            next_note_state, next_dur_state = notes_duration_tuples(part[i + 1:i + self.order + 1])
            update_dict(self.note_dict, self.note_updates, cur_note_state, next_note_state)
            update_dict(self.duration_dict, self.duration_updates, cur_dur_state, next_dur_state)
        if len(part):
            # Add EOL after each part
            update_dict(self.note_dict, self.note_updates, cur_note_state, MidiMarkovChain.EOL)
            # We do not add to duration because we want to be able to draw as much as needed from it

        if update:
            if log:  # Verbose?
                print("Learned " + ' '.join(part))
            self.calculate_probability()
            self.calculate_cdf()

    @staticmethod
    def toState(chordNote):
        """
        Turns chordnote into and note and duration or split chord chord and duration
        :param chordNote: music21 chordNote
        """
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
        """
        Convert note and duration to a music21 note
        :param note: Note in a format C5
        :param duration: Duration of a note as a number
        :return: Music21 note
        """
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
        """
        Convert notes and chords into a music21 stream
        :param notesAndChords: Notes and chords with a durations
        :return: Music21 stream
        """
        stream = music21.stream.Stream()
        piano = music21.stream.Part()
        piano.insert(music21.instrument.Flute())
        for note in notesAndChords:
            piano.append(note)
        stream.append(piano)
        return stream

# EOF
