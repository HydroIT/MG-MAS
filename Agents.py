from creamas.core import CreativeAgent, Environment, Simulation, Artifact
from mg_mas import MidiMarkovChain
from mas_memory import ListMemory

class MarkovMidiAgent():
    def __init__(self, env, markov_chain, composer_index, name=None, generation_attempts=10,
                 memory_size=20):
        """
        MarkovMidiAgent is a class responsible for generating MIDI songs based on the Markov chains calculated from MIDI files. Also deals with evaluation which is based on likelihood and novelty of the artifact.
        :param env: enviroment object where is agent called from
        :param markov_chain: markov_chain object with calculated transitions and probabilities of notes and durations
        :param composer_index: integer that represents which composer was chosen
        :param name: optional - name of the agent
        :param generation_attempts: optional - how many artifact to generate for evaluation, default=10
        :param memory_size: optional - how many artifacts to memorize in the ListMemory object, default 20
        """
        if name is None:
            super().__init__(env)
        else:
            super().__init__(env, name=name)
        self.mmc = markov_chain
        self.env = env
        self.composer_index = composer_index
        self.generation_attempts = generation_attempts
        self.mem = ListMemory(memory_size)

    def evaluate(self, artifact):
        """
        Method that evaluates the artifact, based on it's likelihood and novelty.
        :param artifact: generated MIDI notes and durations from generate() function.
        Returns float number between 0-1 (the bigger the better evaluation).
        """
        likelihood = self.mmc.likelihood(artifact)
        novelty = self.novelty(artifact)
        evaluation = (likelihood + novelty) / 2
        return evaluation

    def generate(self):
        """
        Method calling generate_piece() from MidiMarkovChain class which returns a MIDI tones with durations
        """
        midi = list(self.mmc.generate_piece(length=30))
        return midi

    def invent(self):
        """
        Invent a number of artifacts (self.generation attempts) and evaluate them. Only the artifact with best evaluation score is returned.
        """
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
        return best

    def novelty(self, artifact):
        """
        Calculate novelty of the artifact based on the memory. If created notes are different from the set of memorized ones the difference score increases and afterwards the novelty is higher.
        :param artifact: generated MIDI notes and durations from generate() function.
        """
        novelty = 1.0
        diff = 0
        #print(artifact)
        memory = self.mem.artifacts
        for i,n in enumerate(artifact):
            for memart in memory:
                try:
                    if n != memart[i]:
                        diff += 1
                except:
                    pass
        print(len(memory))
        diff /= len(artifact)
        if len(memory) > 0:
            diff /= len(memory)
        return diff

    async def act(self):
        """
        Agents act by inventing new songs and then memorize them in the memory
        """
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
