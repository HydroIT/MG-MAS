# MG-MAS

This is the homepage for a Computational Creativity (CC) course project that deals with Music Generation from Midi files.
Team members are Lukas Vozda, Nikolaus Risslegger and Idan Tene.

# Requirements
The code is built to run on Python 3.5.2 (or higher) and uses the following libraries:
- creamas (Creative Multi Agent Systems library)
- keras (Machine Learning library, depends in turn on Numpy, Theano, Scipy, etc)
- h5py (Library to handle HDF5 files)
- music21 (MIDI file and music parsing library)

# Files
- Agents.py - Includes implentation of the different agents
- NotesMC.py - Markov Chain model that learns notes and durations respectively, and imposes some structure on the generated output
- DurationMC.py - Markov Chain model that learns only durations from given MIDI streams
- MidiLSTM.py - LSTM Neural Network that generates a sequence of notes\rests
- JudgeLSTM.py - LSTM Neural Network that evaluates a sequence of notes\chords\rests
- MG.py - Generates the agents, loads their respective weights\Markov Chains models, and runs the environment

# Installation

# How to run the code

