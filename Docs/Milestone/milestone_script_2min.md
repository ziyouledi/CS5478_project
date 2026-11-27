# Milestone presentation script (2 min)

Speaker notes in the deck are generated from this file: edit here, then re-run `build_milestone_slides.py`.

**Slide 08** (~10 s)

In short, the whole pipeline now works: a trained expert, the intent signals, and the first Stage 1 students, so let me go straight to what I found.

**Slide 09** (~50 s)

On the left, each curve is a student learning the task. Blind gets no intent; Wrist-2 is also told where the expert wants to move the wrist. First finding: Seed matters a lot. We trained the same Blind student three times with different random seeds. To reach 80 percent of the expert, one run needed 1,500 iterations, another 3,200. So an intent only counts as helpful if it is clearly faster than that range, and Wrist-2 is not; all three Wrist-2 land inside it.

Second finding, on the right: for human hands, two synergies explain over 80 percent of grasps; for the expert I trained, only 36 percent. But the human data are static grasps. Turning a cube needs fingers that move independently, and even the expert's smooth, measured joint motion is just as high-dimensional. So a small hand intent may help less than I hoped.

**Slide 10** (~40 s)

Three challenges. First, that high dimensionality. I still run the hand intents as planned; if a few dimensions help little, that is itself a finding, and my second task, a simpler rotation, will test whether it compresses better.

Second, Isaac Lab's built-in success count is a running average that starts at zero, so the same policy can read 3 or 16 depending on how long you run it. I use my own evaluator that simply counts the successes in every episode.

Third, the seed noise you just saw: I keep three seeds per condition and add more where results are close.

**Slide 11** (~10 s)

Next, I plan to finish Stage 1 by this week, then hide the goal in Stage 2, ahead of the demo in 3 weeks. Thank you.
