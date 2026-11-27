# Final presentation script (10 min)

Speaker notes in the deck are generated from this file: edit here, then re-run `build_final_slides.py`.
About 1,250 words at ~130 words per minute. Timings in brackets add up to 9:55 (12 slides).

**Slide 01** (~15 s)

Hi, we are Team 34. Our question is: how little does a dexterous robot hand need to be told? We measure what a low-dimensional intent should contain, and how often it must be updated.

**Slide 02** (~55 s)

Many robot systems are now hierarchical. A high-level planner, such as a language model, a teleoperator, or a brain–computer interface, decides what to do. A fast low-level controller moves the fingers. The two are linked by an interface that carries only a few numbers, often slowly: a language model can take seconds per decision.

So, given a controller that runs at 60 hertz, which few numbers should the planner send? Finger synergies, object motion, or the goal itself? How often must they arrive? And how much delay and precision loss can they tolerate? Prior systems pick one interface and never measure this budget. We measure it on in-hand cube reorientation with a Shadow Hand.

**Slide 03** (~70 s)

Our method is teacher–student. First we train an expert with PPO that sees the goal orientation. Then we train students with the goal hidden. Instead of the goal, each student gets an intent of k numbers computed from the expert or the task. Hand-k is the expert's next action, projected onto its top k principal components. Wrist-2 is just the two wrist commands. Obj-k describes where the cube should turn next. Goal-4 is the goal quaternion itself.

Then we degrade the channel. A zero-order hold refreshes the intent only every N steps, from 60 hertz down to 2 hertz, and the student is trained under that rate. We also add delay and quantisation at test time.

Our metric is Recovery: zero means a student with no intent at all, one means the expert.

**Slide 04** (~45 s)

Everything runs in Isaac Lab on NUS SoC Slurm GPUs: 2,048 parallel hands per run, 5,000 PPO iterations, and an asymmetric critic that still sees the goal. We trained about 170 student runs, each with 3 seeds, and evaluated every saved checkpoint on 1,024 episodes, counting successes per episode.

Two engineering lessons. Isaac Lab's built-in success metric is a running average that understates performance, so we wrote our own evaluator. And restoring a checkpoint bit-exactly still restarts the learning-rate schedule, which we checked before extending training.

**Slide 05** (~65 s)

Here is the demo, two tasks side by side: on the left, any target orientation; on the right, rotation about the vertical axis. First, the expert, which sees the goal; the floating cube is the target. Next, the goal is hidden and there is no intent. On task one the hand holds the cube but cannot know where to turn it. On task two it learns a trick: it keeps spinning the cube and passes through each goal.

Back to task one. Hand-8 at 60 hertz recovers almost all of the expert's skill. The same eight numbers at 10 hertz, with a student trained at that rate, still turn the cube, but reach far fewer goals. Goal-4 at 60 hertz works, and at 2 hertz it still works, just more slowly: it reaches each new goal one refresh late.

Last, task two with the yaw goal. At 60 hertz it turns back and forth to each target. At 2 hertz it waits on a stale goal and falls below the spinning trick.

**Slide 06** (~60 s)

First finding. With the goal visible, no intent made learning faster. Seven conditions were statistically indistinguishable, ANOVA p equal to 0.82. With the goal hidden, the content matters a lot. Hand-8 recovers 0.87 of the expert, and more components help up to about eight. The object's next rotation, obj-1, recovers nothing. The two wrist commands recover only 0.29. Studies of human grasping treat the wrist as a separate module, because it barely correlates with the fingers. In our expert, the wrist is as correlated with the fingers as the fingers are with each other, so two raw wrist numbers waste the budget: two whole-hand components recover 0.50. The expert's actions are high-dimensional: eight components explain 81 percent of the variance, while two synergies explain over 80 percent of human grasps.

**Slide 07** (~65 s)

This is the central result. At 60 hertz, hand-8 is the best interface. At 10 hertz it collapses to 0.18, even though the student was trained at that rate. Every action-level intent falls to almost zero by 2 hertz. Goal-4 degrades gracefully: 0.77, 0.71, 0.65, and 0.43 at 2 hertz. From 10 hertz down, goal-4 beats every other interface, with p at most 0.012. The dotted line is a hard ceiling: if the goal is refreshed only every half second, at most one goal can be reached per refresh.

That direction is not surprising. One thing is: the level in between, obj-4, which says where the cube should turn next, is never better than goal-4 at any rate. So why do the interfaces differ so much?

**Slide 08** (~45 s)

On the left, the same data against the update rate: four separate curves, so the rate alone explains little. On the right, we replace the rate by how wrong the held value is: zero means still exact, one means no better than a constant, two means unrelated to the true value. Now the points follow one trend. The expert's action flips sign on a quarter of all steps; the goal changes only when the hand succeeds, about every 0.4 seconds.

One exception, the dotted line: obj-4 and goal-4 change equally fast, but obj-4 loses more. A subgoal drifts with the cube, so its held value is soon outside the success tolerance. So what matters is how much the signal changes between updates, measured against the task tolerance.

**Slide 09** (~55 s)

Latency gives the same picture, with numbers. Hand-8 dies at one tenth of a second of delay, with 91 percent of episodes dropping the cube. Goal-4 loses half its value between half a second and one second, and keeps only 21 percent at two seconds. For precision, eight bits per component is lossless, but four bits keeps only 16 percent, because the step becomes as large as the success tolerance.

So the goal channel has a budget: about 32 bits per goal, and well under one second of latency. A language model that needs several seconds per decision is outside that budget. On this task, the goal level does not fit a slow planner either, because the task itself asks for a new goal every 0.4 seconds.

**Slide 10** (~60 s)

Two checks. Left: is the rate penalty only under-training? We resumed goal-4 runs to 10,000 iterations. 60 and 5 hertz gain about 0.12, but 2 hertz gains only 0.04 and stays far below its ceiling. The gap is real.

Right: our second task, rotation about the vertical axis. A two-number yaw goal works as well as the full quaternion, so the channel only needs the degrees of freedom of the goal. But the blind student already reaches 63 percent of the expert by spinning. And at 2 hertz, both goal intents fall below blind: 16 successes against 20. A stale goal is worse than no goal. Goal-level intents are not robust by default. The action-level intent, hand-8, repeats the first task's pattern here. At 60 hertz it is level with the goal intents, about 37 successes. At 10 hertz it falls behind them, 26 against 29 to 31. At 2 hertz it is the worst interface, 12 successes, below the goal intents and below blind. So the reversal is not specific to one task, though on this simpler motion hand-8 degrades more gently than before.

**Slide 11** (~65 s)

To summarise. We built a reproducible pipeline that measures an intent interface by content, rate, delay, and precision. Our main takeaway: an intent survives a slow channel only while its value stays valid between updates. That gave three results we did not expect. The subgoal level lost to the goal level at every rate. A stale goal can be worse than no goal. And on this task even goals change faster than a language model answers.

Limitations: everything is in simulation, with one expert seed and three student seeds. The goal intent is the hidden goal itself, so we compare interface levels, not ways to compress an expert. And we only tested a zero-order hold: a planner could instead send a short action sequence, or send a goal only when it changes.

Next, we would test those two channels, and replace the oracle with a real slow planner.

**Slide 12** (~10 s)

In short: before you choose an interface, measure how fast its signal goes stale. Thank you, and we are happy to take questions.
