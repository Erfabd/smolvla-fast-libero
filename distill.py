"""Label-free step distillation of SmolVLA's action expert, and the offline checks used in the notebook.

A dataset is a dict of numpy arrays, one row per sample:
    cam1, cam2  (N, 256, 256, 3) uint8   front camera and wrist camera
    tokens      (N, 48) int64            instruction tokens, padded with -1
    state       (N, 32) float32          normalised robot state (zeros for the phone video)
    task        (N,) int                 task id, used to hold tasks out
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.policies.common.vla_utils import resize_with_pad

# Only the action expert and its projections are trained; vision and language stay frozen.
TRAINABLE = ("vlm_with_expert.lm_expert", "action_in_proj", "action_out_proj", "action_time_mlp")


def phone_dataset(path, tokenizer):
    """Phone frames: the oblique 'close' view plays the front camera, the top-down 'front' view the wrist camera."""
    d = np.load(path)
    sentence = dict(zip(d["task_nums"].tolist(), d["sentences"].tolist()))
    tokens = np.full((len(d["task_id"]), 48), -1, dtype=np.int64)
    for i, t in enumerate(d["task_id"].tolist()):
        ids = tokenizer(sentence[t] + "\n")["input_ids"][:48]  # same format LeRobot uses
        tokens[i, : len(ids)] = ids
    return {"cam1": d["close"], "cam2": d["front"], "tokens": tokens,
            "state": np.zeros((len(tokens), 32), np.float32), "task": d["task_id"]}


def sim_dataset(path):
    """Model inputs captured from LIBERO rollouts by fast_eval.py --capture. One task per instruction."""
    s = np.load(path)
    _, task = np.unique(s["tokens"], axis=0, return_inverse=True)
    return {"cam1": s["cam1"], "cam2": s["cam2"], "tokens": s["tokens"], "state": s["state"], "task": task.reshape(-1)}


class Distiller:
    def __init__(self, policy, tokenizer, device="cuda"):
        self.policy, self.model, self.device = policy, policy.model, device
        self.pad_id = tokenizer.pad_token_id
        for p in policy.parameters():
            p.requires_grad_(False)
        self.trainable = {n: p for n, p in self.model.named_parameters() if n.startswith(TRAINABLE)}
        for p in self.trainable.values():
            p.requires_grad_(True)
        self.original = {n: p.detach().cpu().clone() for n, p in self.trainable.items()}

    def load_weights(self, path=None):
        """Load student weights from a file, or restore the original checkpoint when path is None."""
        weights = self.original if path is None else torch.load(path, map_location="cpu")
        with torch.no_grad():
            for n, p in self.trainable.items():
                p.copy_(weights[n])

    def inputs(self, ds, idx):
        idx = np.asarray(idx)
        image = lambda x: resize_with_pad(torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float() / 255,
                                          512, 512, pad_value=0) * 2 - 1
        tok = torch.from_numpy(ds["tokens"][idx]).to(self.device)
        tok = tok[:, : int((tok >= 0).sum(1).max())]
        mask = tok >= 0
        masks = [torch.ones(len(idx), dtype=torch.bool, device=self.device)] * 2
        return ([image(ds["cam1"][idx]), image(ds["cam2"][idx])], masks, tok.masked_fill(~mask, self.pad_id),
                mask, torch.from_numpy(ds["state"][idx]).to(self.device))

    def run(self, ds, idx, noise, steps):
        """Action chunk (B, 50, 32) for the given starting noise and number of denoising steps."""
        self.policy.config.num_steps = steps
        return self.model.sample_actions(*self.inputs(ds, idx), noise=noise.clone()).float()

    @torch.no_grad()
    def teacher_targets(self, ds, repeats=4, batch=8, seed=0):
        """The 10-step model's answer for every sample, from `repeats` different starting noises."""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        noises, targets, frames = [], [], []
        for _ in range(repeats):
            for i in range(0, len(ds["task"]), batch):
                idx = np.arange(i, min(i + batch, len(ds["task"])))
                noise = torch.randn(len(idx), 50, 32, device=self.device, generator=gen)
                noises.append(noise.cpu())
                targets.append(self.run(ds, idx, noise, 10).cpu())
                frames.append(torch.from_numpy(idx))
        return {"noise": torch.cat(noises), "target": torch.cat(targets), "frame": torch.cat(frames)}

    @staticmethod
    def average_targets(t):
        """Same rows as `t`, but every target replaced by the mean of the teacher's answers for that frame."""
        target = t["target"].clone()
        for f in t["frame"].unique():
            rows = t["frame"] == f
            target[rows] = t["target"][rows].mean(0)
        return dict(t, target=target)

    @torch.no_grad()
    def distance_to_teacher(self, ds, t, rows, steps):
        """Relative error to the teacher over the 7 executed action dimensions (0 = identical)."""
        num = den = 0.0
        for i in range(0, len(rows), 16):
            r = rows[i:i + 16]
            out = self.run(ds, t["frame"][r].numpy(), t["noise"][r].to(self.device), steps)
            target = t["target"][r].to(self.device)
            num += (out[..., :7] - target[..., :7]).pow(2).sum().item()
            den += target[..., :7].pow(2).sum().item()
        return (num / den) ** 0.5

    def train(self, ds, t, test_tasks, save_to, steps=1000, lr=1e-5, batch=8, k=1, seed=0):
        """Train the k-step student to reproduce the teacher. Always starts from the original weights."""
        self.load_weights(None)
        is_test = torch.from_numpy(np.isin(ds["task"], test_tasks))[t["frame"]]
        train_rows, test_rows = torch.where(~is_test)[0], torch.where(is_test)[0]
        print(f"train samples: {len(train_rows)}   held-out samples: {len(test_rows)}")
        print(f"before training: {k}-step distance to teacher on held-out tasks = "
              f"{self.distance_to_teacher(ds, t, test_rows, k):.3f}")
        params = list(self.trainable.values())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0)
        gen = torch.Generator().manual_seed(seed)
        start = time.time()
        for step in range(1, steps + 1):
            r = train_rows[torch.randint(0, len(train_rows), (batch,), generator=gen)]
            for group in opt.param_groups:
                group["lr"] = lr * min(1.0, step / 50)  # short warm-up
            out = self.run(ds, t["frame"][r].numpy(), t["noise"][r].to(self.device), k)
            loss = F.mse_loss(out, t["target"][r].to(self.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 100 == 0:
                print(f"step {step:4d}   train loss {loss.item():.4f}   held-out distance "
                      f"{self.distance_to_teacher(ds, t, test_rows, k):.3f}   ({time.time() - start:.0f} s)")
        torch.save({n: p.detach().cpu() for n, p in self.trainable.items()}, save_to)
        print("saved", save_to)

    @torch.no_grad()
    def noise_spread(self, ds, steps, frames, repeats=6, seed=0):
        """Same input, `repeats` different starting noises: how much do the answers differ? (0 = not at all)"""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        outs = []
        for _ in range(repeats):
            part = []
            for i in range(0, len(frames), 16):
                idx = frames[i:i + 16]
                noise = torch.randn(len(idx), 50, 32, device=self.device, generator=gen)
                part.append(self.run(ds, idx, noise, steps)[..., :7])
            outs.append(torch.cat(part))
        x = torch.stack(outs)
        return (x.var(0).mean() / x.mean(0).pow(2).mean()).sqrt().item()
