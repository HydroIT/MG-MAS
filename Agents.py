from creamas.core import CreativeAgent, Artifact
from MidiLSTM import MidiLSTM
from JudgeLSTM import JudgeLSTM
from ListMemory import ListMemory
from DurationMC import DurationMarkovChain
from NotesMC import MidiMarkovChain


# Produces solely based on markov chains
class MarkovMidiAgent(CreativeAgent):
    def __init__(self, env, markov_chain, composer_index, name=None, length=30, generation_attempts=10,
                 memory_size=20, judges=None, learning_threshold=0.6, logger=None):
        """
        MarkovMidiAgent is a class responsible for generating MIDI songs based on the Markov chains calculated
        from MIDI files. Also deals with evaluation which is based on likelihood and novelty of the artifact.
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
        self.mmc = MidiMarkovChain()
        self.mmc.load(markov_chain)
        self.composer_index = composer_index
        self.generation_attempts = generation_attempts
        self.length = length
        self.memory = ListMemory(memory_size)
        self.logger = logger
        self.judges = judges if judges is not None else list()
        self.learning_threshold = learning_threshold
        if self.logger:
            self.logger.debug("Agent {} initialized!".format(self.name))

    def evaluate(self, artifact):
        """
        Method that evaluates the artifact, based on it's likelihood and novelty.
        :param artifact: generated MIDI notes and durations from generate() function.
        Returns float number between 0-1 (the bigger the better evaluation).
        """
        likelihood = self.mmc.likelihood(artifact.obj)
        novelty = self.novelty(artifact)
        evaluation = (likelihood + novelty) / 2
        framing = {'composer': self.composer_index}
        return evaluation, framing

    def generate(self):
        """
        Method calling generate_piece() from MidiMarkovChain class which returns a stream wrapped in an artifact
        """
        midi = Artifact(self, MidiMarkovChain.toStream(self.mmc.generate_piece(length=self.length)))
        return midi

    def invent(self):
        """
        Invent a number of artifacts (self.generation attempts) and evaluate them.
        Only the artifact with best evaluation score is returned.
        Artifacts that are highly valued by judge agents are learned as well.
        """
        best = None
        max_value = 0
        for i in range(0, self.generation_attempts):
            midi = self.generate()
            eval, fr = self.evaluate(midi)
            if eval > max_value:
                best = midi
                max_value = eval
                best.add_eval(self, eval, fr=fr)
                for judge in self.judges:
                    v, n, s = judge.gauge(midi, self.composer_index)
                    temp_score = (v + n + s) / 3
                    if temp_score > self.learning_threshold:
                        self.learn(midi, self.composer_index)
                        if self.logger:
                            self.logger.debug("Agent {} learned his own generated music".format(self.name))
        if self.logger:
            self.logger.debug("Markov Agent {} invented {} with score {}".format(self.name, best.obj, max_value))
        return best

    def novelty(self, artifact):
        """
        Calculate novelty of the artifact based on the memory.
        If created notes are different from the set of memorized ones the difference score increases and
        afterwards the novelty is higher.
        :param artifact: generated MIDI sequence notes and durations from generate() function.
        """
        score = 0.0
        memory = self.memory.artifacts
        artifact = list(artifact.obj.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
        for memart in memory:
            memnotes = list(memart.sorted.flat.getElementsByClass(["Note", "Chord", "Rest"]))
            for n1, n2 in zip(artifact, memnotes):
                p1 = MidiMarkovChain.toState(n1)[0]  # Disregard duration
                p2 = MidiMarkovChain.toState(n2)[0]  # Disregard duration
                if p1 != p2:
                    score += 1
        score /= len(artifact)
        if len(memory) > 0:
            score /= len(memory)
        return score

    def learn(self, artifact, _):
        """
        Learns a given artifact, ignores the composer index (shared across all agents in our environment)
        """
        if self.logger:
            self.logger.debug("Agent {} learned given artifact".format(self.name))
        self.mmc.easy_learn(artifact.obj)
        self.memory.memorize(artifact.obj)

    async def act(self):
        """
        Agents act by inventing new songs and then memorize them in the memory
        """
        artifact = self.invent()
        self.env.add_candidate(artifact)


# Produces notes based on LSTM and duration based on MC
class MarkovLSTMAgent(CreativeAgent):
    def __init__(self, env, weights, pickled_mc, composer_index, judges, name,
                 memory_size=20, generation_sequence=100, generation_length=None, learning_threshold=0.5,
                 logger=None):
        super().__init__(env, name=name)  # Generate an agent with a name
        self.model = MidiLSTM(composer_index)
        self.composer_index = composer_index
        self.memory = ListMemory(memory_size)
        self.gen_sequence = generation_sequence
        self.generation_length = generation_length
        self.judges = judges
        self.model.load_weights(weights)
        self.mc = DurationMarkovChain()  # Duration markov chain
        self.mc.load(pickled_mc)  # Load previously generated data
        self.learning_threshold = learning_threshold
        self.logger = logger
        if logger:
            logger.debug("Agent {} initialized!".format(name))

    def evaluate(self, artifact):
        """
        Evaluates the given artifact by asking judge agents and using own agents novelty value
        """
        score = 0.0
        for judge in self.judges:
            v, n, s = judge.gauge(artifact, self.composer_index)
            temp_score = (v + n + s) / 3
            if temp_score > self.learning_threshold:
                self.learn(artifact, None)
            score += temp_score
        score /= len(self.judges)
        score = (score + self.novelty(artifact)) / 2
        fr = {'composer': self.composer_index}
        if self.logger:
            self.logger.debug("{} evaluated {}'s artifact with {}".format(self.name, artifact.creator, score))
        return score, fr

    def learn(self, artifact, _):
        """
        Learns a given artifact, ignores the composer index (shared across all agents in our environment)
        """
        if self.logger:
            self.logger.debug("Agent {} learned given artifact".format(self.name))
        self.model.train_single(artifact.obj)
        self.mc.easy_learn(artifact.obj)
        self.memory.memorize(MidiLSTM.stream2inputs(artifact.obj))

    def invent(self):
        """
        Invents an artifact
        """
        notes = self.model.generate(iterations=self.gen_sequence,
                                    sequence_length=self.generation_length)  # Generate note from LSTM model
        # Generates duration and zip with notes
        result = Artifact(self, MidiLSTM.to_midi_stream(notes, self.mc.generate(len(notes))))
        score, fr = self.evaluate(result)
        result.add_eval(self, score, fr=fr)
        if self.logger:
            self.logger.debug("Markov-LSTM Agent {} invented {} with score {}".format(self.name, result.obj, score))
        return result

    def novelty(self, artifact):
        # Compare notes to memorized items (disregard duration)
        score = 0.0
        notes = MidiLSTM.stream2inputs(artifact.obj)  # Convert artifact to just notes
        for mem in self.memory.artifacts:  # Memory holds lists of midi values
            for n1, n2 in zip(notes, mem):
                if n1 != n2:
                    score += 1
            score /= len(notes)
        n_artifacts = len(self.memory.artifacts)
        if n_artifacts > 0:
            return score / n_artifacts
        return score

    async def act(self):
        # Create, vote, memorize, etc
        artifact = self.invent()
        self.env.add_candidate(artifact)


# Does not produce, only judges given note sequences
class JudgeAgent(CreativeAgent):
    def __init__(self, env, weights, name, value_threshold=0.25, learning_iterations=2, logger=None):
        super().__init__(env, name=name)  # Generate an agent with a name
        self.model = JudgeLSTM()
        self.model.load_weights(weights)
        self.value_threshold = value_threshold
        self.li = learning_iterations
        self.logger = logger
        if logger:
            self.logger.debug("Agent {} initialized!".format(self.name))

    def gauge(self, artifact, composer_index):
        """
        Gauges a given artifact (with composer index property) and returns value, surprisingness and novelty
        """
        probs = self.model.predict(artifact.obj)
        value = max(probs) - self.value_threshold  # value is never 1
        value = 0 if value < 0 else value
        if composer_index is not None:
            novelty = 1.0 - probs[composer_index]
            probs[composer_index] = 0
            surprisingness = max(probs)
        else:
            novelty = 0
            surprisingness = 0
        if self.logger:
            self.logger.debug("{} gave {}'s artifact the following scores:\n\tValue: {}"
                              "\n\tNovelty: {}"
                              "\n\tSurprisingness: {}".format(self.name, artifact.creator, value, novelty,
                                                              surprisingness))
        return value, novelty, surprisingness

    def evaluate(self, artifact):
        """
        Attempts to evaluate a given artifact. If no composer is given in the framing,
        ignores novelty and surprisingness and divides value by 2.
        Returns the composer framing as well (or none)
        """
        try:
            index = artifact.framings['composer']
        except:
            index = None
        fr = {'composer': index}
        v, n, s = self.gauge(artifact, index)
        if index:
            score = (v + n + s) / 3
        else:
            score = v / 2
        return score, fr

    def learn(self, artifact, composer_index):  # Judge learns winning artifact
        self.model.fit_single(artifact.obj, composer_index, self.li)

    async def act(self):
        pass  # Do nothing

#EOF