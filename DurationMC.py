import music21
import random
import pickle


class DurationMarkovChain:
    alpha = 0.12  # Coefficient for probability smoothing

    def __init__(self, music21stream=None, order=1):
        """
        Initializes a markov chain from given raw text.
        :param music21stream: a music file where the probabilities are learnt from
        :param order: Order of markov chain.
        """
        self.order = order
        self.duration_dict = dict()
        self.duration_probs = dict()
        self.duration_cdfs = dict()
        self.duration_updates = list()  # Efficiently update only relevant keys
        if music21stream is not None:
            self.easy_learn(music21stream)
            self.update_all()

    def save(self, filename):
        if len(self.duration_updates) > 0:
            self.update_all()
        with open(filename, "wb") as f:
            pickle.dump(self.duration_dict, f)
            pickle.dump(self.duration_probs, f)
            pickle.dump(self.duration_cdfs, f)
            pickle.dump(self.order, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.duration_dict = pickle.load(f)
            self.duration_probs = pickle.load(f)
            self.duration_cdfs = pickle.load(f)
            self.order = pickle.load(f)

    def generate(self, length=40, start_duration=None):
        """
        Generate the Markov chain with the given length
        """
        if start_duration is None:  # No start state, randomly-uniformly select one (usage of set promises uniformity)
            start_duration = random.choice(list(set(self.duration_cdfs.keys())))
        elif start_duration not in self.duration_dict.keys():  # Bad start state given
            raise LookupError('Cannot find start token in state transitions dictionary - "{}"'.format(start_duration))

        # Generate durations using the created CDFs
        prev_duration = start_duration  # Keep track of previous state
        dur = list(start_duration)
        while len(dur) < length:
            rnd = random.random()  # Random number from 0 to 1
            cdf_dur = self.duration_cdfs[prev_duration]
            cp_dur = cdf_dur[0][1]
            i = 0
            # Go through the cdf_dur until the cumulative probability is higher than the random number 'rnd'.
            while cp_dur < rnd:
                i += 1
                cp_dur = cdf_dur[i][1]
            dur.append(cdf_dur[i][0][-1])
            if cdf_dur[i][0] in self.duration_cdfs.keys():
                prev_duration = cdf_dur[i][0]
        return dur

    def _calculate_probability(self):
        """
        Calculate probability with respect to update list
        """
        diff_values = len(set(self.duration_dict.keys()))  # Total different states (smoothes probabilities)
        for key in self.duration_updates:  # Only update via the list
            if key not in self.duration_probs:
                self.duration_probs[key] = dict()
            sub_dict = self.duration_dict[key]
            total = sum(sub_dict.values()) + diff_values * DurationMarkovChain.alpha
            for w, c in sub_dict.items():
                self.duration_probs[key][w] = (float(c) + DurationMarkovChain.alpha) / total

    def _calculate_cdf(self):
        """
        Calculates CDF based on updated duration list
        """
        for key in self.duration_updates:  # Only update via the update list
            if key not in self.duration_cdfs:  # New key to CDF dictionary
                self.duration_cdfs[key] = dict()
            items = self.duration_probs[key].items()  # Get probabilities for updated key
            # Sort the list by the second index in each item and reverse it from highest to lowest.
            sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
            cdf = []  # Calculate CDF
            cumulative_sum = 0.0
            for c, prob in sorted_items:
                cumulative_sum += prob
                cdf.append([c, cumulative_sum])
            cdf[-1][1] = 1.0  # For possible rounding errors
            self.duration_cdfs[key] = cdf

    def update_all(self):
        """
        Simply updates the probabilities and CDFs and clears the update list
        """
        self._calculate_probability()
        self._calculate_cdf()
        self.duration_updates.clear()

    def easy_learn(self, stream):
        """
        Learns a given stream by formatting it as needed.
        """
        data = list(stream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))  # Extract notes, chords, rests
        data = DurationMarkovChain.get_durations(data)  # get the durations
        self.learn(data)  # Learn thew new durations
        self.learn(list(reversed(data)))  # learn also backwards data

    @staticmethod
    def get_durations(notes):
        """
        Extract the durations of a given list of notes
        """
        durations = list()
        for note in notes:
            try:
                if note.duration.quarterLength > 0:  # remove 0 length
                    durations.append(note.duration.quarterLength)
            except:  # Skip too small durations as well (they raise an error)
                continue
        return durations

    def learn(self, part, update=False, log=False):
        """
        Learn durations from list of notes
        :param part: List of notes
        :param update: Should the probabilities and CDFs be updated on this learning run (default false)
        :param log: Verbose or not (default false)
        """
        for i in range(len(part) - self.order - 1):
            cur_state = tuple(part[i: i + self.order])  # Get current state
            next_state = tuple(part[i + 1: i + self.order + 1])  # Get next state
            if cur_state not in self.duration_dict:  # New state for transition dictionary
                self.duration_dict[cur_state] = dict()
            if next_state not in self.duration_dict[cur_state]:  # New next state for transition dictionary
                self.duration_dict[cur_state][next_state] = 1
            else:
                self.duration_dict[cur_state][next_state] += 1  # Exists :)
            if cur_state not in self.duration_updates:  # Add to update list
                self.duration_updates.append(cur_state)

        if update:  # Should update?
            if log:  # Verbose
                print("Learned " + ' '.join(part))
            self.update_all()


# if you want test it locally uncomment it!
'''
bachs = music21.corpus.getBachChorales()
mc = DurationMarkovChain(music21.corpus.parse(bachs[0]), order=6)

for i in range(1, int(len(bachs)/12)):  # 100 samples should be enough for now (studying both reverse and normal!)
    print("Learning bach #" + str(i + 1))
    mc.easy_learn(music21.corpus.parse(bachs[i]))

print("Calculating probabilities & CDFs...")
mc.update_all()
print("Updated probs and cdfs")

chain = mc.generate(length=50)
print(chain)
mc.save("bach_duration.mc")
'''
