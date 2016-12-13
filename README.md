# MG-MAS

MG-MAS (music generation - multi agent system) is developed as a project for the Computational Creativity and Multi-Agen Systems lecture in fall 2016. The developers are international students who are taking a stay abroad at the University of Helsinki:

* Idan Tene
* Lukas Vozda
* Nikolaus Risslegger

The project tries to generate pleasent sounding pieces of music and its main targets are

- Plan A: write a program which creates automatically awesome music, get famous, get rich, don't have to work the entire life
- Plan B: at least 5 credits.

## Requirements
The code is built to run on Python 3.5.2 (or higher) and uses the following libraries:

- creamas (Creative Multi Agent Systems library)
- keras (Machine Learning library, depends in turn on Numpy, Theano, Scipy, etc)
- h5py (Library to handle HDF5 files)
- music21 (MIDI file and music parsing library)

In certain circumstances also follwing libraries have to be installed (these should be automatically installed with keras):

- numpy
- scipy
- theano

In addition, sometimes changing the definition file for keras is required (so it uses Theano and not TensorFlow).
Please check with version fits at best for your machine.
To achieve this, open `~/.keras/keras.json` (if it isn't there, you can create it).
The contents of `keras.json` should like like so:
```json
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```


## Files and Structure
- `Agents.py` - Includes implentation of the different agents
- [`NotesMC.py`](https://github.com/HydroIT/MG-MAS#markov-chain) - Markov Chain model that learns notes and durations respectively, and imposes some structure on the generated output
- [`DurationMC.py`](https://github.com/HydroIT/MG-MAS#duration-markov-chain) - Markov Chain model that learns only durations from given MIDI streams
- [`MidiLSTM.py`](https://github.com/HydroIT/MG-MAS/blob/master/README.md#midi---lstm-midilstmpy) - LSTM Neural Network that generates a sequence of notes\rests
- [`JudgeLSTM.py`]() - LSTM Neural Network that evaluates a sequence of notes\chords\rests
- `MG.py` - Generates the agents, loads their respective weights\Markov Chains models, and runs the environment


## How to run the code
**have no idea what to write - how the final enviroment works**
```python
python3 open file?!
```


## Explanations of the Code

###Markov Chain (`mg_mas.py`)
Generate two Markov chains, one for the pitches, a second for the durations, analyzing the same songs but independent to each other.
Merge the two chains into a set of Music21 notes.
Put some repetitions (intro, chords, outro or similar) to the generated piece to let it sound a bit structurated.
Create several tracks, calculate the liklihood and export the piece with the hightest result.
The class consists of a dictionary, probabilities and cdfs for the pitch and the durations respectively.
To seperate the tones (which might be chords, single tones and rests) and durations a quite nasty piece has to be used:
```python
@staticmethod
def toState(chordNote):
    if chordNote.isNote:
        return chordNote.nameWithOctave, chordNote.duration.quarterLength
    elif chordNote.isChord:
        return tuple([note.nameWithOctave for note in chordNote]), chordNote.duration.quarterLength
    elif chordNote.isRest:
        return "Rest", chordNote.duration.quarterLength
    else:
        return MidiMarkovChain.EOL
```
During the init process those variables are filled with the values calculated of the given musicial datasets, those functions are based on the provided example for the exercises during the lecture. As splitted into the two parts those functions are adapted for working with the note objects or the durations.
```python
def calculate_cdf(self):
    def calc_cdf(dictionary, cdfs):
        ...

def calculate_probability(self):
    def calc_prob(dictionary, probs, updates):
        ...
```

The creating of the new *piece* of music happens in the `generate_piece` method where new durations and tones are generated.
```python
def generate(self, length=40, start_note=None, start_duration=None):
    ...
while len(gen) < length and prev_note != MidiMarkovChain.EOL:
    rnd = random.random()  # Random number from 0 to 1
        cdf_note, cdf_dur = self.note_cdfs[prev_note], self.duration_cdfs[prev_duration]
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
        # same for duration
```

**The try to put structure into the generation**
One thought was to modify the creation a bit towards an actual rhythm. Simplest way would be to generate loops and let them appear more often (f.e. A-B-A-B-B-C). 
But the dark side of putting that much control into the generation is, that the program cannot be that creative by itself. Therefore that idea was rejected soon. The last scraps of that idea are found in the function `getSongWithRepeatings(merged_notes)`.


### Duration Markov chain (`DurationMC.py`)
Due to the problem that `MidliLSTM.py` didn't produce different lengths of the tones we had to implement a workaround to get different length.
Basically it has the same logic as the "big brother" `NotesMC.py` - but less complexity due the zipping and unzipping of the notes and duration can be ignored.
The learn (and also others) function might be easier to understand without the in- and out-wrapping:
```python
def learn(self, part, update=False, log=False):
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
```           

###MIDI - LSTM (`MidiLSTM.py`)
After the not satisfying results of the Markov Chains we thought it might be a good idea to generate the train an agent during using deep long short-term memory (LSTM) networks.

The whole calculation is time and/or resources-consuming, but differs on the given input. An average training with takes about 15-120 seconds each epoch. 

The most important function is **PLEASE IDAN probably say what it does**
```python
 def train(self, epochs=100, n=20):
	 x_raw, raw_length = MidiLSTM._get_notes_from_composer(self.composer_idx, n)  # Get data
     x_train = list()
     y_train = list()
     for sample in x_raw:  # Refactor training data to sequences of #timesteps
         x, y = MidiLSTM._sample2sequences(sample, self.timesteps)
         x_train += x
         y_train += y
     x, y = self._reshape_inputs(x_train, y_train)
     self.model.fit(x, y, nb_epoch=epochs, batch_size=self.batch_size, callbacks=self.callbacks_list)
```

###Judge LSTM (`JudgeLSTM.py`)
An attempt at neural-network-based evaluation function for midi files.
Final result gives a probability for each "genre" from the following list:

	0. corpus.getComposer('bach') 
	1. corpus.getComposer('beethoven')
	2. corpus.getComposer('essenFolksong')
	3. corpus.getComposer('monteverdi')
	4. corpus.getComposer('oneills1850')
	5. corpus.getComposer('palestrina')
	6. corpus.getComposer('ryansMammoth')
	7. corpus.getComposer('trecento')

The probabilities can then be used as evaluating all the different aspects of creativity:

- **value**: Does this NN consider the midi stream to be > 0.5 for any "genre"?
- **novelty**: How far is the score given by the NN to the agent's idea of his genre? (i.e. the loss function?)
- **surprisingness**: Incorporating the agents data, if he hasn't seen anything from genre 4 but got the highest score there, then quite surprising? etc...

This network does not take into account the length\duration, but can be extended to such quite easily.

**IDAN WHAT IS IMPORTANT?**



## Results
Firstly, the first outputs haven't conviced us that our generated music will find many sympathizers. The tracks may have some chords which sound familar but the whole creation has no passion - like creating tension and their dissolution. 
Chords and rests are so unlikely to happen in the output caused of the few successive occurs in the input:
if a more rests would exist they are combined to one longer one - therefore the Markov chain slightly ignores them.

So, songs without any feelings, no rests (also a chord is a small mircacle) - and alltogether sounds like a five year old child found a piano and hits into the keys.
What we could do better - composing the songs starting with the basic chors (for exampe the often used I-V-vi-IV combination) and adding afterwards fitting notes and their durations.
Not to use any bars (rhythmic beats) also minimizes the chance of getting a well-hearted song. 
But to introduce this (and of course many others) would cost quite a bit time and explode the workload.

Second, as we figured out, the best value for the length of the Markov chain is around 6. 


