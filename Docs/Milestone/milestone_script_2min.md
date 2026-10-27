# Milestone presentation script (2 min)

**Slide 08** (~26 s)

Since the proposal, the whole pipeline runs end to end. The goal-aware expert is trained, 164 million transitions, and solves about 26 cube reorientations per episode. From 1.2 million of its actions we built the PCA basis, and every intent type is implemented and verified. Stage 1 is now running on the cluster: four of the 21 runs are trained and evaluated.

**Slide 09** (~38 s)

Two early results. On the left, learning curves relative to the expert. Blind uses the expert's exact setup with new seeds, so its seed spread is our noise floor; any intent has to beat it. Wrist-2 has one seed so far, so treat it as a preview. On the right, a surprise: two synergies explain over 80 percent of human grasps, but our RL expert needs 14 dimensions for 95 percent, because it acts almost bang-bang. So the real test is whether a few low-variance directions still carry the useful information.

**Slide 10** (~29 s)

Three main challenges. First, that bang-bang expert: we keep the PCA intent as planned and report the result, with smoothed targets as a fallback. Second, Isaac Lab's success metric is a running average that reads 3 or 16 at the same reward, so we built our own per-episode evaluator. Third, compute: each run takes about one and a half GPU-hours on a shared cluster, so everything runs as throttled Slurm arrays.

**Slide 11** (~18 s)

Next, we finish Stage 1 by October 31, run Stage 2 with the goal hidden until November 10, then one extension before the November 18 demo. Priority one is a clean Stage 1 answer; priority two, the Stage 2 recovery curve. Thank you.
