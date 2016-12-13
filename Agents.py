from creamas.core import CreativeAgent, Environment, Simulation, Artifact
from MidiLSTM import MidiLSTM
from JudgeLSTM import JudgeLSTM
from ListMemory import ListMemory
from DurationMC import DurationMarkovChain
import numpy as np

# Produces solely based on markov chains
class MarkovMidiAgent(CreativeAgent):
    def __init__(self, env, markov_chain, composer_index, name=None, generation_attempts=20,
                 memory_size=20):
        if name is None:
            super().__init__(env)
        else:
            super().__init__(env, name=name)  # Generate an agent with a name
        pass

    def evaluate(self, artifact):
        # Compute likelihood of artifact using underlying markov chain
        pass

    def generate(self):
        # Generate a note sequence based on markov chain
        pass

    def invent(self):
        # Generate generation_attempts sequences and choose the best one according to self evaluation
        pass

    def novelty(self, artifact):
        # Use some memory bank and see how different it is from recently-known artifacts
        # (Per memory sequence: number of notes that are different, divide by total number of notes)
        # (Then divide by size of memory bank)
        # Can also use judge agents?
        pass

    async def act(self):
        # Create, vote, memorize, etc...
        pass


# Produces notes based on LSTM and duration based on MC
class MarkovLSTMAgent():#CreativeAgent):
    def __init__(self, env, weights, pickled_mc, composer_index, judges, name=None,
                 memory_size=20, generation_sequence=1000):
        # if name is None:
        #     super().__init__(env)
        # else:
        #     super().__init__(env, name=name)  # Generate an agent with a name
        self.model = MidiLSTM(composer_index)
        self.composer_index = composer_index
        print("Markov LSTM Agent initialized")
        self.memory = ListMemory(memory_size)
        self.gen_sequence = generation_sequence
        self.judges = judges
        self.model.load_weights(weights)
        print("Weights loaded")
        self.mc = DurationMarkovChain()  # Duration markov chain
        self.mc.load(pickled_mc)  # Load previously generated data
        print("Markov Chain Dictionaries loaded")

    def evaluate(self, artifact):
        score = 0.0
        for judge in self.judges:
            v, n, s = judge.gauge(artifact.obj, self.composer_index)
            score += judge.evaluate(artifact.obj)
        return score / len(self.judges)

    def invent(self, length=None):
        notes = self.model.generate(sequence_length=length)  # Generate note from LSTM model
        self.memory.memorize(notes)  # memorize notes
        return notes, self.mc.generate(len(notes))  # Generates duration and attach to notes

    def novelty(self, artifact):
        # Compare notes to memorized items (disregard duration)
        score = 0.0
        for mem in self.memory.artifacts:
            for n1, n2 in zip(artifact.obj, mem.obj):
                if n1 != n2:
                    score += 1
            score /= len(artifact.obj)
        return score / self.memory.capacity

    async def act(self):
        # Create, vote, memorize, etc
        pass


# Does not produce, only judges given note sequences
class JudgeAgent():#CreativeAgent):
    def __init__(self, env, weights, name=None, value_threshold=0.25, learning_iterations=2):
        # if name is None:
        #     super().__init__(env)
        # else:
        #     super().__init__(env, name=name)  # Generate an agent with a name
        self.model = JudgeLSTM()
        print("Judge agent initialized")
        self.model.load_weights(weights)
        print("Weights loaded")
        self.value_threshold = value_threshold
        self.li = learning_iterations

    def gauge(self, artifact, composer_index):
        probs = self.model.predict(artifact.obj)
        value = max(probs) - self.value_threshold  # value is never 1
        value = 0 if value < 0 else value
        novelty = 1.0 - probs[composer_index]
        probs[composer_index] = 0
        surprisingness = max(probs)
        return value, novelty, surprisingness

    def learn(self, artifact, composer_index):
        self.model.fit_single(artifact.obj, composer_index, self.li)

#EOF

judge = JudgeAgent(None, "weights-classify-70-0.3340.hdf5")
test = MarkovLSTMAgent(None, "weights-composer-0-96-1.7315.hdf5", "bach_duration.mc", 0, [judge])
notes, dur = test.invent()
print(list(zip(notes, dur)))
stream = MidiLSTM._to_midi_stream(notes, dur)
print(judge.gauge(stream, 0))

