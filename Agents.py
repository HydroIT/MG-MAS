from creamas.core import CreativeAgent, Environment, Simulation, Artifact

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

#EOF
