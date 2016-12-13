from creamas.core import CreativeAgent, Environment, Simulation, Artifact
from mg_mas import MidiMarkovChain
from mas_memory import ListMemory
import music21

# Produces solely based on markov chains
class MarkovMidiAgent():
    def __init__(self, markov_chain, composer_index, name=None, generation_attempts=10,
                 memory_size=20, env=None):
        # if name is None:
        #     super().__init__(env)
        # else:
        #     super().__init__(env, name=name)  # Generate an agent with a name
        self.mmc = markov_chain
        self.env = env
        self.composer_index = composer_index
        self.generation_attempts = generation_attempts
        self.mem = ListMemory(memory_size)

    def evaluate(self, artifact):
        likelihood = self.mmc.likelihood(artifact)
        novelty = self.novelty(artifact)
        evaluation = (likelihood + novelty) / 2
        print("Likelihood: ", likelihood, " Novelty: ", novelty)
        return evaluation

    def generate(self):
        midi = list(self.mmc.generate_midi(length=30, save=False))
        return midi

    def invent(self):
        # Generate generation_attempts sequences and choose the best one according to self evaluation
        best = list()
        lkhds = []
        max = 0.00000000000000000000000000000000000000000000
        for i in range (0, self.generation_attempts):
            midi = self.generate()
            eval = self.evaluate(midi)
            lkhds.append(eval)
            print(eval)
            if eval > max:
                best = midi
                max = eval
        print("Saving the best midi with evaluation of: ", max)
        print("All likelihoods:", lkhds)
        #MidiMarkovChain.safe_stream(MidiMarkovChain.toStream(best))
        return best

    def novelty(self, artifact):
        # Use some memory bank and see how different it is from recently-known artifacts
        # (Per memory sequence: number of notes that are different, divide by total number of notes)
        # (Then divide by size of memory bank)
        # Can also use judge agents?
        novelty = 1.0
        diff = 0
        memory = self.mem.artifacts
        for i,n in enumerate(artifact):
            for memart in memory:
                if n != memart.obj[0]:
                    diff += 1
        print(len(memory))
        diff /= len(artifact)
        if len(memory) > 0:
            diff /= len(memory)
        return diff

    async def act(self):
        artifact = self.invent()
        self.mem.memorize(artifact)


# Produces notes based on LSTM and duration based on MC
class MarkovLSTMAgent(CreativeAgent):
    def __init__(self, env, weights, markov_chain, composer_index, judges, name=None,
                 memory_size=20, generation_sequence=1000):
        if name is None:
            super().__init__(env)
        else:
            super().__init__(env, name=name)  # Generate an agent with a name
        pass

    def evaluate(self, artifact):
        # Evaluate using judge agents
        pass

    def invent(self):
        # Create notes from LSTM
        # Create matching duration from Markov Chain
        pass

    def novelty(self, artifact):
        # Compare notes to memorized items (disregard duration)
        pass

    async def act(self):
        # Create, vote, memorize, etc
        pass

# Does not produce, only judges given note sequences
class JudgeAgent(CreativeAgent):
    def __init__(self, env, weights, name=None):
        if name is None:
            super().__init__(env)
        else:
            super().__init__(env, name=name)  # Generate an agent with a name
        pass

    def evaluate(self, artifact):
        # Run given artifact through own LSTM and give predictions
        pass

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
mma = MarkovMidiAgent(mc, 0)
midi = mma.invent()
