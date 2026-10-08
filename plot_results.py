"""Build results/table.md and results/speed_vs_success.png from the evaluation files in results/."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

RUNS = [  # file stem, label, uses distilled weights, x nudge so overlapping intervals stay readable
    ("reference_10step", "10 steps, re-plan every step", False, 1.0),
    ("fast_1step_chunk10", "1 step, re-plan every 10 steps", False, 0.93),
    ("student_sim_1step_chunk10", "student (sim frames)", True, 1.0),
    ("student_phone_1step_chunk10", "student (my phone video)", True, 1.07),
]
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
COLOR = {False: "#2563eb", True: "#d97706"}

results = Path("results")
latency = json.loads((results / "latency.json").read_text())
rows = []
for stem, label, student, nudge in RUNS:
    o = json.loads((results / f"{stem}.json").read_text())["overall"]
    lat = latency[stem]
    rows.append({"label": label, "student": student, "success": o["pc_success"], "ci": o["pc_success_ci95"],
                 "n": o["n_episodes"], "k": o["n_success"], "call_ms": lat["ms_per_call"],
                 "step_ms": lat["ms_per_call"] / lat["n_action_steps"], "nudge": nudge})

lines = ["| configuration | model ms per call | model ms per control step | success | 95% interval |",
         "|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['label']} | {r['call_ms']:.0f} | {r['step_ms']:.1f} | {r['k']}/{r['n']} ({r['success']:.1f}%) "
                 f"| {r['ci'][0]:.0f} to {r['ci'][1]:.0f}% |")
(results / "table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
for r in rows:
    c = COLOR[r["student"]]
    x = r["step_ms"] * r["nudge"]
    ax.errorbar(x, r["success"], yerr=[[r["success"] - r["ci"][0]], [r["ci"][1] - r["success"]]],
                fmt="o", ms=8, color=c, ecolor=c, elinewidth=1.5, capsize=0, alpha=0.95,
                markeredgecolor="white", markeredgewidth=2)
    left = r["step_ms"] > 100
    ax.annotate(r["label"], (x, r["success"]), xytext=(-9 if left else 9, -3), textcoords="offset points",
                fontsize=8, color=INK, ha="right" if left else "left")
for student, name in [(False, "original weights"), (True, "distilled student")]:
    ax.plot([], [], "o", color=COLOR[student], label=name)
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
