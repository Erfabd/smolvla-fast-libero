"""lerobot-eval with three additions.

1. The vision-language part of SmolVLA runs in float16 (the checkpoint ships in bfloat16,
   which a T4 GPU runs slowly).
2. --student=PATH loads distilled action-expert weights on top of the checkpoint.
3. --capture=PATH saves every model input (both cameras, tokens, state) to an .npz file.

It also times every model call. Any other flag goes to lerobot-eval unchanged:

    python fast_eval.py --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero \
        --env.task=libero_spatial --eval.batch_size=1 --eval.n_episodes=4 \
        --policy.num_steps=1 --policy.n_action_steps=10 --output_dir=runs/fast
"""

import sys
import time

import numpy as np
import torch
import lerobot.scripts.lerobot_eval as lerobot_eval


def pop_option(name):
    for arg in sys.argv:
        if arg.startswith(name + "="):
            sys.argv.remove(arg)
            return arg.split("=", 1)[1]


STUDENT = pop_option("--student")
CAPTURE = pop_option("--capture")
make_policy = lerobot_eval.make_policy
call_ms, captured = [], []


def small(img):
    """(1, 3, 512, 512) in [-1, 1] -> (256, 256, 3) uint8, the same format as the phone frames."""
    x = torch.nn.functional.interpolate(img.float(), size=256, mode="area")
    return ((x[0].permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()


def make_fast_policy(*args, **kwargs):
    policy = make_policy(*args, **kwargs)
    policy.model.vlm_with_expert.vlm.to(torch.float16)
    if STUDENT:
        weights = torch.load(STUDENT, map_location="cpu")
        _, unexpected = policy.model.load_state_dict(weights, strict=False)
        assert not unexpected, unexpected
        print(f"[fast_eval] loaded {len(weights)} student tensors from {STUDENT}", flush=True)
    print(f"[fast_eval] num_steps={policy.config.num_steps}  n_action_steps={policy.config.n_action_steps}", flush=True)

    sample = policy.model.sample_actions

    def timed(*a, **kw):
        if CAPTURE:
            images, _, tokens, _, state = a[:5]
            padded = np.full(48, -1, dtype=np.int64)
            padded[: tokens.shape[1]] = tokens[0].cpu().numpy()
            captured.append((small(images[0]), small(images[1]), padded, state[0].float().cpu().numpy()))
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = sample(*a, **kw)
        torch.cuda.synchronize()
        call_ms.append((time.perf_counter() - start) * 1000)
        return out

    policy.model.sample_actions = timed
    return policy


lerobot_eval.make_policy = make_fast_policy

if __name__ == "__main__":
    try:
        lerobot_eval.main()
    finally:
        if captured:
            cam1, cam2, tokens, state = (np.stack(x) for x in zip(*captured))
            np.savez_compressed(CAPTURE, cam1=cam1, cam2=cam2, tokens=tokens, state=state)
            print(f"[fast_eval] captured {len(captured)} model inputs -> {CAPTURE}", flush=True)
        if len(call_ms) > 3:  # the first calls include warm-up
            print(f"[fast_eval] model: {np.median(call_ms[3:]):.0f} ms per call   calls: {len(call_ms)}", flush=True)
