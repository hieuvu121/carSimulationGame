"""Generate slide figures (300 dpi PNG) from the repo's real data and results.

Run from the repo root:
    python -m presentation_tools.make_figures

Uses presentation_tools/output/facts.json (created by compute_facts.py; it is run
automatically if missing). Writes into presentation_figures/. Figures whose source
data does not exist in the repo are skipped with a message.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import RESULTS_DIR, ROOT_DIR  # noqa: E402

FACTS = ROOT_DIR / "presentation_tools" / "output" / "facts.json"
OUT = ROOT_DIR / "presentation_figures"
LABELS = ["fist", "v_sign", "open_palm", "none"]
NICE = {"fist": "Fist", "v_sign": "V sign", "open_palm": "Open palm", "none": "None",
        "no_hand": "No hand"}
COLORS = {"fist": "#2a78d6", "v_sign": "#2fa37a", "open_palm": "#e08a1e", "none": "#8a8f98",
          "no_hand": "#d64545"}

plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote presentation_figures/{name}")


def skip(name, reason):
    print(f"  SKIPPED {name}: {reason}")


def class_distribution(f):
    d = f["dataset"]
    splits = [("Train\n" + ", ".join(d["train_users"]), d["per_class_train"]),
              ("Test\n" + ", ".join(d["test_users"]), d["per_class_test"])]
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(splits))
    for lab in LABELS:
        vals = np.array([s[1][lab] for s in splits])
        ax.bar([s[0] for s in splits], vals, bottom=bottom, color=COLORS[lab],
               label=NICE[lab], width=0.55, edgecolor="white")
        for i, v in enumerate(vals):
            ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center", color="white",
                    fontsize=15, fontweight="bold")
        bottom += vals
    ax.set_ylabel("Samples")
    ax.set_title("Samples per class and split (user-based split)")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "01_class_distribution_per_split.png")


def confusion(f):
    t = f["test"]
    cm = np.array(t["confusion_matrix"], float)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=14, color="white" if norm[i, j] > 0.5 else "black")
    names = [NICE[l] for l in t["confusion_labels"]]
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix (row-normalised)\n"
                 f"users {', '.join(f['dataset']['test_users'])}, accuracy {t['accuracy']:.1%}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "02_confusion_matrix_normalised.png")


def per_class_f1(f):
    t = f["test"]["per_class"]
    vals = [t[l]["f1"] for l in LABELS]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar([NICE[l] for l in LABELS], vals, color=[COLORS[l] for l in LABELS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=16)
    ax.axhline(f["test"]["macro_f1"], ls="--", color="black", lw=1.5)
    ax.text(3.45, f["test"]["macro_f1"] + 0.015, f"macro {f['test']['macro_f1']:.2f}",
            ha="right", fontsize=14)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 score (test set)")
    ax.set_title("Per-class F1 on held-out test users")
    save(fig, "03_per_class_f1.png")


def grid_heatmap(f):
    if "grid" not in f:
        return skip("04_validation_heatmap.png", "no grid results (run compute_facts without --no-grid)")
    rows = f["grid"]["rows"]
    Cs = sorted({r["C"] for r in rows})
    kernels = ["linear", "rbf"]
    m = np.array([[next(r["mean"] for r in rows if r["C"] == c and r["kernel"] == k)
                   for c in Cs] for k in kernels])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    for i in range(len(kernels)):
        for j in range(len(Cs)):
            best = (kernels[i] == f["grid"]["best_params"]["svm__kernel"]
                    and Cs[j] == f["grid"]["best_params"]["svm__C"])
            ax.text(j, i, f"{m[i, j]:.3f}" + (" ★" if best else ""), ha="center", va="center",
                    fontsize=16, color="white" if m[i, j] < m.mean() else "black")
    ax.set_xticks(range(len(Cs)), [f"C={c:g}" for c in Cs])
    ax.set_yticks(range(len(kernels)), kernels)
    ax.set_ylabel("Kernel (gamma='scale')")
    ax.set_title("Leave-one-user-out validation accuracy\n(train users u01–u04; grid from train.py)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "04_validation_heatmap_C_x_kernel.png")


def latency_breakdown():
    files = sorted(RESULTS_DIR.glob("latency_*.csv"))
    rows = []
    for p in files:
        with p.open() as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        return skip("05_latency_breakdown.png",
                    "no results/latency_*.csv from real webcam runs (Evan: run main --log-latency)")
    stages = ["landmarks_ms", "predict_ms", "dispatch_ms", "response_ms"]
    med = [np.median([float(r[s]) for r in rows]) for s in stages]
    p95 = [np.percentile([float(r[s]) for r in rows], 95) for s in stages]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stages))
    ax.bar(x - 0.2, med, 0.4, label="median")
    ax.bar(x + 0.2, p95, 0.4, label="p95")
    ax.set_xticks(x, ["MediaPipe", "features+SVM", "smooth+dispatch", "end-to-end"])
    ax.set_ylabel("ms")
    ax.set_title(f"Latency per stage ({len(rows)} actions)")
    ax.legend(frameon=False)
    save(fig, "05_latency_breakdown.png")


def offline_accuracy_by_group(f):
    t = f["test"]
    groups = [("User", t["per_user"]), ("Condition", t["per_condition"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 4]})
    for ax, (title, d) in zip(axes, groups):
        keys = list(d)
        vals = [d[k]["accuracy"] for k in keys]
        bars = ax.bar(keys, vals, color="#2a78d6" if title == "User" else "#2fa37a")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", fontsize=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Test accuracy per {title.lower()}")
        ax.tick_params(axis="x", rotation=0 if title == "User" else 15)
    axes[0].set_ylabel("Accuracy (offline, recorded test frames)")
    save(fig, "06_offline_test_accuracy_per_user_condition.png")


def live_success():
    p = RESULTS_DIR / "live_trials.csv"
    if not p.exists():
        return skip("06b_live_success_rate.png",
                    "no results/live_trials.csv (live gesture trials per user/condition not recorded)")


def raw_vs_smoothed(f):
    r = f["replay"]
    seq = r["sequence"]
    order = ["no_hand", "none", "open_palm", "v_sign", "fist"]
    y = {k: i for i, k in enumerate(order)}
    t = np.array([s["t"] for s in seq])
    fig, ax = plt.subplots(figsize=(15, 6))
    # Shade the true label of each recorded burst.
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i]["true"] != seq[start]["true"]:
            ax.axvspan(t[start], t[i - 1] + 0.1, color=COLORS[seq[start]["true"]], alpha=0.10)
            ax.text((t[start] + t[i - 1]) / 2, len(order) - 0.35, f"true: {NICE[seq[start]['true']]}",
                    ha="center", fontsize=14)
            start = i
    ax.scatter(t, [y[s["raw"]] + 0.12 for s in seq], s=14, color="#8a8f98",
               label="raw prediction (after 0.70 threshold)")
    ax.step(t, [y[s["stable"]] - 0.12 for s in seq], where="post", color="black", lw=2.2,
            label="smoothed (0.25 s majority vote)")
    ev = [(s["t"], s["event"]) for s in seq if s["event"]]
    for k, (te, tok) in enumerate(ev):
        ax.axvline(te, color=COLORS[tok], lw=2, ls="--",
                   label="command fired" if k == 0 else None)
    ax.set_yticks(range(len(order)), [NICE[o] for o in order])
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel("Time (s) — rows replayed at the 10 Hz recording rate")
    ax.set_title(f"Raw vs smoothed predictions — offline replay of {r['file']} (test user)")
    ax.legend(loc="lower right", frameon=True, fontsize=13)
    save(fig, "07_raw_vs_smoothed_replay.png")


def main():
    if not FACTS.exists():
        from presentation_tools.compute_facts import main as compute
        compute([])
    f = json.loads(FACTS.read_text())
    print("Generating figures:")
    class_distribution(f)
    confusion(f)
    per_class_f1(f)
    grid_heatmap(f)
    latency_breakdown()
    offline_accuracy_by_group(f)
    live_success()
    raw_vs_smoothed(f)


if __name__ == "__main__":
    main()
