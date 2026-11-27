# Milestone presentation script (2 min)

Speaker notes in the deck are generated from this file: edit here, then re-run `build_milestone_slides.py`.

**Slide 08** (~10 s)

The full pipeline is now running: a trained expert, the intent signals, and the first Stage 1 students. Let me show the early results.

**Slide 09** (~50 s)

On the left, each curve shows a student learning the task. Blind gets no intent; Wrist-2 gets the expert's target wrist movement. Seed matters. I trained the same Blind student three times with different random seeds. One run reached 80 percent of the expert in 1,500 iterations; another took 3,200. An intent only helps if it is clearly faster than that range. Wrist-2 does not clear it: all three runs are inside the range.

On the right, two synergies explain over 80 percent of human grasps, but only 36 percent for the expert I trained. The human data are static grasps. Turning a cube needs fingers to move independently, and even the expert's smooth, measured joint motion is high-dimensional. So a small hand intent may help less than I hoped.

**Slide 10** (~40 s)

Three challenges. First, high dimensionality. I will run the hand intents as planned. If a few dimensions help little, that is a finding. My second task, a simpler rotation, will test whether it compresses better.

Second, Isaac Lab's built-in success count is a running average that starts at zero, so the same policy can read 3 or 16 depending on how long it runs. My evaluator counts successes in every episode.

Third, seed noise. I keep three seeds per condition and add more when results are close.

**Slide 11** (~10 s)

I plan to finish Stage 1 this week, then hide the goal in Stage 2 ahead of the demo in three weeks. Thank you.
