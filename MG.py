import logging
from creamas.core import Environment, Simulation
from Agents import *
import datetime

# Logging setup. This is simplified setup, the logger is being passed to different agents for printouts
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class MidiEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote(self, age):
        artifacts = self.perform_voting(method='IRV')
        if len(artifacts) > 0:
            accepted = artifacts[0][0]
            value = artifacts[0][1]
            composer = accepted.framings[accepted.creator]['composer']  # Get the artifacts composer index
            logger.info("Vote winner by {} (composer={}, val={})".format(accepted.creator, composer, value))
            accepted.obj.show('midi')
            fn = 'export-' + str(datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")) + '.mid'
            accepted.obj.write('midi', fp=fn)
            print("MIDI saved: {}".format(fn))
            for agent in self.get_agents(address=False):
                agent.learn(accepted, composer)
        else:
            logger.info("No vote winner!")

        self.clear_candidates()

env = MidiEnvironment.create(('localhost', 5555))
judge = JudgeAgent(env, r"weights\weights-classify-70-0.3340.hdf5", name="Judge #1", logger=logger)
MarkovLSTMAgent(env, r"weights\weights-composer-0-99-1.8424.hdf5", r"weights\bach_duration.mc", composer_index=0,
                judges=[judge], name="Agent #1", logger=logger)
MarkovMidiAgent(env, r"weights\bach.mc", name="MC #1", logger=logger, composer_index=0, judges=[judge])

sim = Simulation(env, log_folder='logs', callback=env.vote)
sim.async_steps(10)
sim.end()

# EOF
