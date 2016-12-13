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
judges = list()
judges.append(JudgeAgent(env, r"weights\weights-classify-70-0.3340.hdf5", name="Judge #1", logger=logger))
judges.append(JudgeAgent(env, r"weights\weights-classify-90-0.4151.hdf5", name="Judge #2", logger=logger))
'''
composers = {0: 'bach', 1: 'beethoven', 2: 'essenFolksong', 3: 'monteverdi',
                 4: 'oneills1850', 5: 'palestrina', 6: 'ryansMammoth', 7: 'trecento'}
'''
markov_lstm_pairs = {0: (r"weights\weights-composer-0-99-1.8424.hdf5", r"weights\bach_duration.mc"),
                     1: (r"weights\weights-composer-1-10-3.3988.hdf5", r"weights\beethoven_duration.mc"),
                     2: (r"weights\weights-composer-2-32-2.2389.hdf5", r"weights\essenFolksong_duration.mc"),
                     3: (r"weights\weights-composer-3-88-1.9254.hdf5", r"weights\monteverdi_duration.mc"),
                     4: (r"weights\weights-composer-4-17-2.2279.hdf5", r"weights\oneills1850_duration.mc"),
                     5: (r"weights\weights-composer-5-07-2.4109.hdf5", r"weights\palestrina_duration.mc"),
                     6: (r"weights\weights-composer-6-98-1.8839.hdf5", r"weights\ryansMammoth_duration.mc"),
                     7: (r"weights\weights-composer-7-98-1.6597.hdf5", r"weights\trecento_duration.mc")}
mc_pairs = {0: r"weights\bach.mc",
            1: r"weights\beethoven.mc",
            2: r"weights\essenFolksong.mc",
            3: r"weights\monteverdi.mc",
            4: r"weights\oneills1850.mc",
            5: r"weights\palestrina.mc",
            6: r"weights\ryansMammoth.mc",
            7: r"weights\trecento.mc"}
for k, v in markov_lstm_pairs.items():
    MarkovLSTMAgent(env, v[0], v[1], composer_index=k, judges=judges, name="LSTM #" + str(k), logger=logger)
for k, v in mc_pairs.items():
    MarkovMidiAgent(env, v, name="MC #" + str(k), logger=logger, composer_index=k, judges=judges)

sim = Simulation(env, log_folder='logs', callback=env.vote)
sim.async_steps(10)
sim.end()

# EOF
