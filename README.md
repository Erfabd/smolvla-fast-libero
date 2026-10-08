# Making SmolVLA fast on a T4, and what I learned from my phone videos

SmolVLA (`HuggingFaceVLA/smolvla_libero`) on LIBERO-Spatial, Franka Panda arm, one Colab T4.

I wanted to see how fast SmolVLA could run on a T4 without retraining it, and whether my phone videos could make the 1-step version as good as the 10-step one.

On LIBERO-Spatial, I got the model from 842 ms per call down to about 17 ms per control step by switching to float16, using 1 denoising step, and re-planning every 10 steps. The fast version got 30/40 successes (75%), compared with 28/40 (70%) for the original 10-step setup.

I also tried distilling the model using 20 simple manipulation tasks that I recorded with my phone. The result was worse than the original model. I first thought the problem was the domain gap between my videos and LIBERO, but a control experiment on LIBERO frames gave the same problem. That led me to look more closely at what the 1-step model was actually learning.

That look pointed at the training target, so I changed it: instead of one teacher sample, the student learns the teacher's average answer. With that change my phone video did as well as LIBERO's own frames (55% each, up from 37.5% for the phone student). But every student I distilled was still below the original 1-step model.

| configuration (float16 vision-language model) | model ms per call | model ms per control step | success (40 episodes) | 95% interval |
|---|---|---|---|---|
| 10 denoising steps, re-plan every step | 586 | 586 | 28/40 (70%) | 55 to 82% |
| **1 step, re-plan every 10 steps** | 172 | **17** | **30/40 (75%)** | 60 to 86% |
| student on LIBERO frames, single-sample target | 174 | 17 | 21/40 (52.5%) | 37 to 67% |
| student on my phone video, single-sample target | 173 | 17 | 15/40 (37.5%) | 24 to 53% |
| student on LIBERO frames, average target | 171 | 17 | 22/40 (55%) | 40 to 69% |
| student on my phone video, average target | 172 | 17 | 22/40 (55%) | 40 to 69% |

The checkpoint as shipped (bfloat16, 10 steps, re-plan every step) takes 842 ms per call on the same GPU.
Times are medians measured inside the LIBERO runs, which happened on different Colab machines. Re-measured in a
single session on random inputs: 854 ms as shipped, 704 ms in float16 with 10 steps, and 153 to 193 ms with 1 step
(two measurements of the same computation, so roughly 20% is machine noise), which is 15 to 19 ms per control step.

![speed against success](results/speed_vs_success.png)

| 1 step, re-plan every 10 steps | 10 steps, re-plan every step | phone student, single-sample target |
|---|---|---|
| ![](media/fast_success.gif) | ![](media/reference_success.gif) | ![](media/student_phone_failure.gif) |

## 1. Making it fast

I timed the model first. A LIBERO step took 1.3 s; about 0.5 s of that is the simulator, the rest is the model.
Inside one model call (random inputs, T4):

| stage | bfloat16 (as shipped) | float16 |
|---|---|---|
| vision encoder, 2 images | 213 ms | 26 ms |
| prefix pass (images, text, state) | 76 ms | 69 ms |
| flow-matching loop, 10 steps | 639 ms | 640 ms |

I used three simple changes, without retraining the model:

1. **float16 instead of bfloat16.** The T4 does not have a fast bfloat16 path. This made the vision encoder much faster and also gave a smaller error compared with full precision (0.0025 vs 0.031).

2. **1 denoising step instead of 10.** One step takes about 60 ms.

3. **Re-plan every 10 control steps.** The model predicts 50 actions at a time, but the original setup only executes one of them before running the model again.

## 2. Using my phone video

**Data.** 20 one-handed tasks with household objects (cup, bowl, fork, knife, banana, orange), each filmed from
above and from the side. 24 time-matched frames per clip, 480 frame pairs. The side view plays LIBERO's front camera
and the top view its wrist camera. Instructions are in [`data/tasks.txt`](data/tasks.txt).

![my frames](media/phone_frames.png)

### Idea

My phone videos don't contain action labels, so I couldn't train the policy directly from them. Instead, I used the original 10-step model as a teacher.

For each frame, I ran the teacher and used its output as the target for a 1-step student. The hope was that the student would learn to produce the same action in one step, giving me most of the quality of the original model at a much lower cost.

### Result

The student got closer to the teacher on the offline metric: the distance dropped from 0.67 to 0.35.

But this did not translate to better control. On LIBERO, success dropped from 75% for the fast original model to 37.5% for the phone-video student.

### Control

I wanted to check whether the phone videos were actually the problem. So I captured 593 LIBERO frames during the fast evaluation, including the real robot state, and trained the same student on those frames.

The result was similar: the student got closer to the teacher offline (0.49 → 0.28), but LIBERO success still dropped, this time to 52.5%.

So the main problem seems to be the distillation setup itself. The phone videos add a domain gap on top of that, which makes things even worse.

### Why

I think the main problem is what the 1-step model is being asked to learn.

With flow matching, one Euler step from noise tends to give something close to the mean action for an observation. That output is fairly stable across different starting noises. With 10 steps, the model produces a sample, so the starting noise matters much more.

I tested this by running each model on the same input with 6 different starting noises. The original 1-step model stayed relatively stable, while both students became much more sensitive to the noise.

| model | LIBERO frames | phone frames |
|---|---|---|
| original, 1 step | 0.15 | 0.20 |
| student (LIBERO frames), 1 step | 0.54 | 0.54 |
| student (phone video), 1 step | 0.58 | 0.64 |
| teacher, 10 steps | 0.59 | 0.68 |

The students became as noise-dependent as the teacher while still being far from it: they lost the stable mean and
did not gain an accurate sample. That matches the videos I watched: the students reach the bowl, then hesitate and
jitter at the grasp. The teacher is just as noise-dependent and still succeeds, so this number alone does not predict failure.

The phone frames gave the same diagnosis as the LIBERO frames, in the same order and at similar values.

### Fix: train on the teacher's average

If the target was the problem, the student should learn the mean directly. For each frame I averaged the teacher's
answers from its 4 starting noises and trained the same student on that average. Nothing else changed.

| student, average target | noise spread, 1 step (original model) | distance to target, held-out tasks | LIBERO success |
|---|---|---|---|
| my phone video | 0.19 (0.20) | 0.51 → 0.39 | 22/40 (55%) |
| LIBERO frames | 0.16 (0.15) | 0.33 → 0.31 | 22/40 (55%) |

Two things came out of this:

- **My phone video caught up with LIBERO's own frames.** Both students kept the original model's stability, and the phone student went
  from 37.5% to 55%, the same score as the student trained on LIBERO's own frames. On its own this gain is not
  significant at 40 episodes (Fisher's exact test, p = 0.18), but the 15-point gap between phone video and LIBERO
  frames from the first recipe is gone.
- **Distillation still hurt.** Across all four students, 80 of 160 episodes succeeded, against 30 of 40 for the
  original 1-step model (Fisher's exact test, p = 0.005). The LIBERO-frame student barely moved offline
  (0.33 → 0.31) and still lost 20 points. Small changes to the action expert that my offline measurements do not
  see are enough to hurt in closed loop. I have not found out which part of the action chunk they affect.

## What worked, and what didn't

Worked:
- Measuring before optimising. The bfloat16 problem only showed up because I timed each stage.
- The control run. Without it I would have blamed the phone video.
- Using the diagnosis to change the target. With the average target my phone video was as useful as LIBERO's own
  frames.

Did not work:
- Distillation in every form I tried: two targets, two data sources, four students, all below the original 1-step
  model.
- My phone data was very different from LIBERO. I filmed forks, bananas and a knife on a white
  table, while LIBERO uses a dark bowl, a white plate and a wooden table. With the first recipe the
  phone student was 15 points below the student trained on LIBERO frames; with the average target the
  gap disappeared, but neither student beat the original. If I did this again, I'd record the actual
  LIBERO task with the same objects and camera views.
- Distance to the teacher as an offline metric. It ranked the students above the original model; LIBERO ranked
  them below.
- Ten-episode evaluations. They gave 10/10 for the fast setting and 2/10 for the phone student; 40 episodes gave
  75% and 37.5%. The first initial state of each task is easy. Those runs are in [`results/early`](results/early).
- Several of my own predictions: that the model was 95% of step time (it is 57%), that the student
  would help when executing 10 actions at once (it got worse), and that a student that keeps the
  original's stability would match it (it did not).

## Why I set it up this way

- **LIBERO-Spatial:** I used the same setting as the challenge and it fits on a T4.
- **Timing inside the simulator:** `fast_eval.py` measures every model call during evaluation. The numbers from a separate benchmark did not match what I saw inside the actual loop.
- **Model time instead of wall-clock time:** About 0.5 s of each LIBERO step comes from physics and rendering, so I don't count that as model time.
- **Only train the action expert:** That is where the compute goes, so I kept the vision and language parts frozen.
- **Hold out tasks:** Phone tasks 17–20 and LIBERO tasks 8–9 are never used for training.
- **40 episodes:** I use 40 episodes and Wilson intervals for the success rates I compare.

## Limitations

- I only tested LIBERO-Spatial, one GPU type, and 40 episodes per configuration. With this setup, differences smaller than about 20 points are hard to take too seriously.
- The LIBERO students were evaluated on the same initial states they were trained on, which gives them an advantage. They still performed worse.
- The average target uses only 4 teacher samples per frame, so it is a noisy estimate of the mean.
- For phone frames, I used the mean robot state from the dataset.
- T4 timings vary by about 10% between Colab machines.

## Run it

Open [`notebook.ipynb`](notebook.ipynb) in Colab with a T4. Sections 1 and 5 take minutes; each LIBERO evaluation
takes 1 to 2 hours and each student about 15 minutes to train. Student weights (about 400 MB each) are not in the
repository; the notebook rebuilds them.

| file | content |
|---|---|
| `notebook.ipynb` | everything, in order, with my results under each step |
| `fast_eval.py` | `lerobot-eval` with float16, student weights, input capture and timing |
| `distill.py` | datasets, teacher targets, student training, offline measurements |
| `plot_results.py` | table and figure from `results/` |
| `results/` | evaluation outputs (`*.json`), model timings, table, figure |
| `data/` | task list and my frames (`phone_frames.zip`, 480 pairs at 256 px) |
