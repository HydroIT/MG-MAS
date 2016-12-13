  .. _markov:


***************
Markov Chains
***************

Idea
=============================
Generate two Markov chains, one for the pitches, a second for the durations, analyzing the same songs but independent to each other.

Merge the two chains into a set of Music21 notes.

Put some repetitions (intro, chords, outro or similar) to the generated piece to let it sound a bit structurated.

Create several tracks, calculate the liklihood and export the piece with the hightest result.


Analyzing
=============================

and go on!


Results
=============================
The first outputs haven't conviced us that our generated music will find many sympathizers. The tracks may have some chords which sound familar but the whole creation has no passion - like creating tension and their dissolution. 
Chords and rests are so unlikely to happen in the output caused of the few successive occurs in the input:
if a more rests would exist they are combined to one longer one - therefore the Markov chain slightly ignores them.

So, songs without any feelings, no rests (also a chord is a small mircacle) - and alltogether sounds like a five year old child found a piano. 
What we could do better - composing the songs starting with the basic chors (for exampe the often used I-V-vi-IV combination) and adding afterwards fitting notes and their durations.
Not to use any bars (rhythmic beats) also minimizes the chance of getting a well-hearted song. 

But to introduce this (and of course many others) would cost quite a bit time and explode the workload.

