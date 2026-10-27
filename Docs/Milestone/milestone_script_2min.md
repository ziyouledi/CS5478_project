# Milestone presentation script (2 min)

Speaker notes in the deck are generated from this file: edit here, then re-run `build_milestone_slides.py`.

**Slide 08** (~10 s)

In short, the whole pipeline now works: a trained expert, the intent signals, and the first Stage 1 students, so let me go straight to what we found.

**Slide 09** (~50 s)

On the left, each curve is a student learning the task. Blind gets no intent; Wrist-2 is also told where the expert wants to move the wrist. First finding: the random seed alone matters a lot. The same Blind setup reaches 80 percent of the expert anywhere between 1,500 and 3,200 iterations. An intent only helps if it beats that spread, and Wrist-2 so far sits inside it.

Second finding, on the right: for human hands, two synergies explain over 80 percent of grasps; for our expert, only 36 percent. But the human data are static grasps. Turning a cube needs fingers that move independently, and even the expert's smooth, measured joint motion is just as high-dimensional. So a small hand intent may help less than we hoped.

**Slide 10** (~40 s)

Three challenges. First, that high dimensionality. We still run the hand intents as planned; if a few dimensions help little, that is itself a finding, and our second task, a simpler rotation, will test whether it compresses better.

Second, Isaac Lab's built-in success number is a running average, so the same policy can read 3 or 16. Our own evaluator counts real successes per episode.

Third, the seed noise you just saw: we keep three seeds per condition and add more where results are close.

**Slide 11** (~10 s)

Next, we finish Stage 1 by October 31, then hide the goal in Stage 2, ahead of the November 18 demo. Thank you.
