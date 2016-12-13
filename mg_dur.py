import music21
import random

class DurationMarkovChain:
    EOL = None  # Universal end of line symbol
    alpha = 0.12  # Coefficient for probability smoothing

    def __init__(self, music21stream, order=1):
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
        self.easy_learn(music21stream)
        self.calculate_probability()
        self.calculate_cdf()

    def generate(self, length=40, start_duration=None):
        """
        Generate the Markov chain with the given length
        """
        if start_duration is None:  # No start state, randomly-uniformly select one (usage of set promises uniformity)
            start_duration = random.choice(list(set(self.duration_dict.keys())))
        elif start_duration not in self.duration_dict.keys():  # Bad start state given
            raise LookupError("Cannot find start token in state transitions dictionary - \"" + str(start_duration) + "\"")

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
            if cdf_dur[i][0] == DurationMarkovChain.EOL:
                return dur  # EOL reached
            dur.append(cdf_dur[i][0][-1])
            if cdf_dur[i][0] in self.duration_cdfs.keys():
                prev_duration = cdf_dur[i][0]
        return dur

    
    def calculate_probability(self):
        def calc_prob(dictionary, probs, updates):
            diff_values = len(set(dictionary.keys()))
            for key in updates:
                if key not in probs:
                    probs[key] = dict()
                sub_dict = dictionary[key]
                total = sum(sub_dict.values()) + diff_values * DurationMarkovChain.alpha
                for w, c in sub_dict.items():
                    probs[key][w] = (float(c) + DurationMarkovChain.alpha) / total

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
        
        # Create CDF
        self.duration_cdfs.clear()
        calc_cdf(self.duration_probs, self.duration_cdfs)
        print("Done calculating durations cdfs...")


    def easy_learn(self, stream):
        """
        Learns a given stream by formatting it as needed.
        """
        data = list(stream.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
        data = self.get_durations(data)
        
        self.learn(data)  
        self.learn(list(reversed(data))) #  learn also backwards data


    def get_durations(self, notes):
        """
        Extract the durations of a given list of notes
        """
        durations = list()
        for note in notes:
        	if note.duration.quarterLength > 0: #  remove 0 length
	            durations.append(note.duration.quarterLength)
        return durations


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

        cur_dur_state = None
        for i in range(len(part) - self.order - 1):
            cur_dur_state = tuple(part[i : i+self.order])
            next_dur_state = tuple(part[i+1 : i + self.order + 1])
            update_dict(self.duration_dict, self.duration_updates, cur_dur_state, next_dur_state)

        if update:
            if log:
                print("Learned " + ' '.join(part))
            self.calculate_probability()
            self.calculate_cdf()