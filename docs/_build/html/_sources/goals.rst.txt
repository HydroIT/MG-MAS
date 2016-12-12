  .. _goals:


***************
Project Goals
***************

Plan A: write a program which generate awesome music, get famous, get rich, don't have to work the entire life

Plan B: at least 5 credits.

General Info
=============================

MG-MAS (music generation - multi agent system) is developed as a project for the Computational Creativity and Multi-Agent Systems lecture in fall 2016. The developers are international students who are taking a stay abroad at the University of Helsinki:

* `Idan Tene <mailto:tene@helsinki.fi>`_

* `Lukas Vozda <mailto:vozda@helsinki.fi>`_

* `Nikolaus Risslegger <mailto:risslegg@helsinki.fi>`_


The project tries to generate pleasent sounding pieces of music, therefore it analyzes existing tracks, generates 
Markov chains and produce a MIDI file::
	
	mg_mas.py

Since the variant using Markov chains was not generating meaningful output a second attempt works with neural networks::

	midi_gen_list.py
