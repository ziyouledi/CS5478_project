# Final presentation script v2 (10 min, draft 2026-11-13)

Draft for the new structure: one 2×1 video per result slide (first 8 s pair A, then 8 s pair B), the old
demo slide removed, set-up slides short. Not yet wired into `build_final_slides.py` (which reads
`final_script_10min.md`). `[A]` / `[B]` mark when each video half starts; speak over the video.
About 1,300 words, ~10:00 at 130 words per minute: no slack left (slides 05–10 reworded 11-13).

**Slide 01** (~10 s)

Hi everyone, we are Team 34. Our core question is: when a slow planner talks to a fast robot hand, what exactly should it say—and how often?

**Slide 02** (~60 s)

With GPT-6 Astra and, last week, Fable 5.2, general-purpose models have started to control robots zero-shot, without a specially trained VLA.

But there is a catch: each decision takes seconds. A dexterous hand needs new commands at 60 hertz. So these models cannot close the loop themselves. They sit on top as a slow planner, and a fast learned controller moves the fingers.

Between the two is a narrow channel. In our project, we model it as a few numbers, sent at a low rate. We call these numbers the intent.

Our question is: which numbers should the planner send, and how often must they arrive for the task to still work?

We study this in simulation, on a Shadow Hand that reorients a cube to a target orientation.

**Slide 03** (~80 s)

Here is our set-up. First, we train an expert that sees the target and solves the task well. Then we train a student that is blind to the target. Why? Because if the student could see it, an intent would be useless. So the only hint the student gets is the intent.

We test intents at three levels.

First, the action level. We call it hand-k. We take the expert's 20 motor commands, two for the wrist and 18 for the fingers, and find the k most common movement patterns of the whole hand. The intent is how much of each pattern to use right now. We also try wrist-2, just the wrist commands, and wrist-2 plus finger-k, which adds the top finger patterns.

Second, the object level: how much further the cube needs to turn, or where it should be a moment from now.

Third, the goal level: the target orientation itself, just four numbers.

Then we make the channel worse: we send the intent less often, late, or with fewer bits.

We score each student from zero percent, no better than having no intent, to a hundred percent, matching the expert.

**Slide 04** (~15 s)

Everything runs in Isaac Lab on the soc GPU cluster: about 180 training runs, each with three to five seeds, and every result is measured on a thousand test episodes.

**Slide 05** (~55 s)

 We use two tasks. On the left, the hand must turn the cube to any 3-D orientation. On the right is a simpler task: it only rotates the cube around the z axis. [A] This is the expert on both. The floating cube demonstrates the target. [B] Now we hide the target from the policy and send no intent. On task one, the hand just holds the cube. So task one cannot be solved without knowing the target. On task two, it finds a trick: it keeps spinning the cube, so sooner or later it passes every target angle. That alone gives 63 percent of the expert's score. So how much the planner needs to say depends on the task.

[If `1127361` supports it: "On task two, two hand patterns already reach X; on task one they reach half the expert."]

**Slide 06** (~55 s)

Now back to task one. Here, the intent is sent at full speed—60 times a second.

[指引看A组] On the left: no intent. On the right: hand-8. With eight hand patterns, the student recovers a massive 87 percent of the expert's performance.

[指引看B组] Now look at the left here: this is hand-2. Just two patterns only recover about half. We found that adding more patterns helps, but the gain flatlines right around eight.

You might wonder: why so many? In human grasping, the Santello team famously showed that just two patterns can explain over 80 percent of hand posture. But our robot expert’s motion is way more complex. For our hand, it takes eight patterns just to reach that 81% mark.

So, what about the other intents? Sending the target itself—just four numbers—scores a 0.77. Sending the remaining turn angle? It gives us basically zero.

The big takeaway here: at full speed, sending eight action numbers performs at least as well as sending the ultimate goal itself."

**Slide 07** (~70 s)

Next, let’s see what happens when we slow the channel down.

[指引看A] At 60 Hertz, hand-8 and the goal perform about the same.

[指引看B] But drop it to 2 Hertz, and look at this: hand-8 crashes to just 3 percent—almost nothing. The goal, however, still holds on to 44 percent. Even at 10 Hertz, the gap is massive: 19 percent for hand-8, versus a solid 71 percent for the goal. We ran this across five seeds, and the goal beat hand-8 every single time at slow rates. Even training twice as long couldn't close this gap.

Now, you might ask: is this only because we hold the commands of a fast expert?

To check this, we trained a new controller. This one can see the target, but it only acts 10 times a second. It reaches just 12 percent of the expert, and at two hertz, nothing.

The takeaway is crystal clear: the hand itself must run on a fast control loop. Only the planner layer above it can afford to be slow."

**Slide 08** (~60 s)

So, why does the goal survive the slow channel while the action crashes? It comes down to one thing: how wrong a message becomes before the next one arrives.

The expert's command to each joint reverses on about one step in four. So if you hold an action, it is soon pushing the wrong way. But the goal? It only changes when the hand actually reaches it, roughly every 0.4 seconds.

[A] Look at this goal intent at two hertz. The message is only wrong in the window between reaching the goal and getting the next update. During that time, the hand simply waits.

[B] Now here is the same motion, slowed down. In orange, we added a waypoint half a radian ahead of the cube, also sent at two hertz. Because the waypoint moves with the cube, it goes stale very fast: it is wrong 81 percent of the time, compared to just 26 percent for the goal.

On the plot, when we measure staleness instead of rate, all the intents follow one trend. The waypoint loses even more, because it quickly drifts outside the success tolerance.

So the takeaway: it is not the rate itself that matters. It is how much the signal changes between updates, compared with the tolerance.

**Slide 09** (~55 s)

And we see the same story with delay.

[A] On the left is hand-8 with zero delay. On the right, we add a delay of just a tenth of a second. The result? The hand fails: it drops the cube in 91 percent of the episodes.

[B] But look at the goal. With that same tenth-of-a-second delay, it keeps most of its performance. Even with a full second of delay, it still holds on to a third.

Beyond time, precision matters too. With eight bits per number, the goal loses nothing. But drop it to four bits? It keeps only 16 percent of its performance. Why? Because at four bits, the rounding error is about as big as our success tolerance.

So, what does this mean for system design? Our goal channel has a budget: about 32 bits per goal, and a delay well under one second. A language model that needs several seconds per decision simply does not fit in this budget.

**Slide 10** (~50 s)

Now, task two adds an interesting twist. Here, the goal is just a single angle, so two numbers are enough.

[A] At sixty hertz, it works well. The goal student reaches 36 goals per episode, clearly beating the 20 we get with no intent.

[B] But drop the rate to two hertz, and something surprising happens: it reaches only 16. Yes, that is worse than having no intent at all!

Why? Remember the spinning trick: with no intent, the student just keeps spinning the cube, which needs no information at all. The goal student, however, has learned to trust the goal. So it chases a target that is already out of date. A stale goal is worse than no goal.

And what about hand-8? It shows the same pattern as in task one. It matches the goal at sixty hertz, but at two hertz it drops to just 12 goals: the worst of all, even below no intent.

**Slide 11** (~60 s)

So, let's bring it all together.

An intent only survives a slow channel if it stays valid between updates.

Actions go stale within a step, so action-level intents need a fast planner. Goals, on the other hand, stay valid longer. They can tolerate a slow planner, but only within a budget. And as we saw in task two, a stale goal can actually do more harm than no goal at all.

And on our task, even the goal changes faster than a language model can answer.

Of course, our work has limitations. It is simulation only, with just one expert and three to five seeds per student. Also, our goal intent is the true goal, an oracle, rather than a planner's guess.

As next steps, we would let the planner send a goal only when it actually changes, or send a short action sequence instead. Most importantly, we would replace our oracle with a real, slow planner.

So before you choose an interface, measure how fast its signal goes stale. Thank you for listening.
