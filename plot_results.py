"""Build results/table.md and results/speed_vs_success.png from the evaluation files in results/."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

RUNS = [  # file stem, table name, plot label, group, x nudge and label offset (points) so overlapping points stay readable
    ("reference_10step", "10 steps, re-plan every step", "10 steps, re-plan every step", "original", 1.0, 0),
    ("fast_1step_chunk10", "1 step, re-plan every 10 steps", "1 step, re-plan every 10 steps", "original", 0.86, 0),
    ("student_sim_1step_chunk10", "student on LIBERO frames, single-sample target", "LIBERO frames", "sample", 0.93, -5),
    ("student_phone_1step_chunk10", "student on my phone video, single-sample target", "my phone video", "sample", 1.0, 0),
    ("student_sim_mean_chunk10", "student on LIBERO frames, average target", "", "average", 1.07, 0),  # shares the next label
    ("student_phone_mean_chunk10", "student on my phone video, average target", "LIBERO frames and my phone video",
     "average", 1.14, 5),
]
LABEL_X = 22  # student labels start here (ms) so they clear the error bars
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
COLOR = {"original": "#2563eb", "sample": "#d97706", "average": "#0d9488"}
GROUPS = [("original", "original weights"), ("sample", "student, single-sample target"),
          ("average", "student, average target")]

results = Path("results")
latency = json.loads((results / "latency.json").read_text())
rows = []
for stem, name, label, group, nudge, dy in RUNS:
    o = json.loads((results / f"{stem}.json").read_text())["overall"]
    lat = latency[stem]
    rows.append({"name": name, "label": label, "group": group, "dy": dy, "success": o["pc_success"], "ci": o["pc_success_ci95"],
                 "n": o["n_episodes"], "k": o["n_success"], "call_ms": lat["ms_per_call"],
                 "step_ms": lat["ms_per_call"] / lat["n_action_steps"], "nudge": nudge})

lines = ["| configuration | model ms per call | model ms per control step | success | 95% interval |",
         "|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['name']} | {r['call_ms']:.0f} | {r['step_ms']:.1f} | {r['k']}/{r['n']} ({r['success']:.1f}%) "
                 f"| {r['ci'][0]:.0f} to {r['ci'][1]:.0f}% |")
(results / "table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
for r in rows:
    c = COLOR[r["group"]]
    x = r["step_ms"] * r["nudge"]
    ax.errorbar(x, r["success"], yerr=[[r["success"] - r["ci"][0]], [r["ci"][1] - r["success"]]],
                fmt="o", ms=8, color=c, ecolor=c, elinewidth=1.5, capsize=0, alpha=0.95,
                markeredgecolor="white", markeredgewidth=2)
    if not r["label"]:
        continue
    if r["group"] == "original":
        left = r["step_ms"] > 100
        ax.annotate(r["label"], (x, r["success"]), xytext=(-9 if left else 9, -3), textcoords="offset points",
                    fontsize=8, color=INK, ha="right" if left else "left")
    else:
        ax.annotate(r["label"], (LABEL_X, r["success"]), xytext=(0, r["dy"] - 3), textcoords="offset points",
                    fontsize=8, color=c, ha="left")
for group, name in GROUPS:
    ax.plot([], [], "o", color=COLOR[group], label=name)
ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
ax.set_xscale("log")
ax.set_xlim(8, 2000)
ax.set_ylim(0, 100)
ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500, 1000]))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("model time per control step, ms (log scale, Colab T4)", color=MUTED, fontsize=9)
ax.set_ylabel("LIBERO-Spatial success, % (40 episodes)", color=MUTED, fontsize=9)
ax.set_title("Speed against success (bars: 95% Wilson intervals; points near 17 ms spread slightly)", loc="left", color=INK, fontsize=10)
ax.grid(True, which="major", color=GRID, linewidth=0.6)
ax.tick_params(which="both", colors=MUTED, labelsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#d1d5db")
fig.tight_layout()
fig.savefig(results / "speed_vs_success.png")
print("wrote results/table.md and results/speed_vs_success.png")
