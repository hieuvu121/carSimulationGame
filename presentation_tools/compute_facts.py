"""Compute presentation facts from the committed data, model and tests.

Run from the repo root:
    python -m presentation_tools.compute_facts            # everything
    python -m presentation_tools.compute_facts --no-grid  # skip the slow LOUO grid re-run

Writes presentation_tools/output/facts.json and prints a readable summary.
Read-only with respect to src/, data/, models/ and results/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from src.app.smoothing import Debouncer, GestureSmoother
from src.capture.preprocess import preprocess, row_to_landmarks
from src.config import (CONFIDENCE_THRESHOLD, LABEL_NONE, MODEL_PATH,
                        RECORD_SAMPLE_HZ, ROOT_DIR, SPLITS_PATH)
from src.data.dataset import load_dataset
from src.data.split import load_splits
from src.model.train import build_pipeline

OUT_DIR = ROOT_DIR / "presentation_tools" / "output"
LABEL_ORDER = ["fist", "v_sign", "open_palm", "none"]


def dataset_facts(dataset, cleaning, splits):
    def per_class(mask):
        c = Counter(dataset.y[mask])
        return {lab: int(c[lab]) for lab in LABEL_ORDER}

    all_mask = np.ones(len(dataset.y), bool)
    train = np.isin(dataset.groups, splits.train_users)
    test = np.isin(dataset.groups, splits.test_users)
    return {
        "rows_read": cleaning.rows_read,
        "rows_kept": cleaning.rows_kept,
        "dropped": cleaning.dropped,
        "excluded_files": cleaning.excluded_files,
        "n_files": int(len(set(dataset.sessions))),
        "users": sorted(set(dataset.groups.tolist())),
        "conditions": sorted(set(dataset.conditions.tolist())),
        "train_users": list(splits.train_users),
        "test_users": list(splits.test_users),
        "per_class_all": per_class(all_mask),
        "per_class_train": per_class(train),
        "per_class_test": per_class(test),
        "handedness_all": dict(Counter(dataset.handedness.tolist())),
        "handedness_per_user": {
            u: dict(Counter(dataset.handedness[dataset.groups == u].tolist()))
            for u in sorted(set(dataset.groups.tolist()))},
    }


def model_test_facts(model, dataset, splits):
    test = np.isin(dataset.groups, splits.test_users)
    X, y = dataset.X[test], dataset.y[test]
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    argmax = np.array(classes)[proba.argmax(1)]
    conf = proba.max(1)
    y_live = np.where(conf < CONFIDENCE_THRESHOLD, LABEL_NONE, argmax)

    p, r, f, s = precision_recall_fscore_support(y, y_pred, labels=LABEL_ORDER, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=LABEL_ORDER)
    pairs = [(LABEL_ORDER[i], LABEL_ORDER[j], int(cm[i, j]))
             for i in range(4) for j in range(4) if i != j]
    pairs.sort(key=lambda t: -t[2])

    def acc_by(keys):
        out = {}
        for k in sorted(set(keys)):
            m = keys == k
            out[k] = {"n": int(m.sum()), "accuracy": float(accuracy_score(y[m], y_pred[m]))}
        return out

    cond = dataset.conditions[test]
    light = np.array([c.split("_")[0] for c in cond])
    dist = np.array([c.split("_")[1] for c in cond])
    lp, lr, lf, _ = precision_recall_fscore_support(y, y_live, labels=LABEL_ORDER, zero_division=0)
    return {
        "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(s[i])}
                      for i, lab in enumerate(LABEL_ORDER)},
        "confusion_labels": LABEL_ORDER,
        "confusion_matrix": cm.tolist(),
        "top_confusions": pairs[:5],
        "per_user": acc_by(dataset.groups[test]),
        "per_condition": acc_by(cond),
        "per_lighting": acc_by(light),
        "per_distance": acc_by(dist),
        "per_hand": acc_by(dataset.handedness[test]),
        "argmax_equals_predict_share": float((argmax == y_pred).mean()),
        "live_threshold": CONFIDENCE_THRESHOLD,
        "live_thresholded_accuracy": float(accuracy_score(y, y_live)),
        "live_thresholded_per_class_recall": {lab: float(lr[i]) for i, lab in enumerate(LABEL_ORDER)},
        "live_rejected_share": float((conf < CONFIDENCE_THRESHOLD).mean()),
        "live_confusion_matrix": confusion_matrix(y, y_live, labels=LABEL_ORDER).tolist(),
        "mean_confidence_correct": float(conf[argmax == y].mean()),
        "mean_confidence_wrong": float(conf[argmax != y].mean()),
    }


def timing_facts(model, dataset, n=1000):
    """Offline per-stage compute time on this machine (no camera, no MediaPipe)."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(dataset.y), size=n, replace=False)
    # Rebuild raw rows: dataset.X is already preprocessed, so time preprocess on raw CSV rows.
    raw_rows = []
    for path in sorted((ROOT_DIR / "data" / "raw").glob("*.csv"))[:4]:
        with path.open() as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                raw_rows.append((np.array(parts[3:-1], float), parts[-1]))
    raw_rows = raw_rows[:n]

    def bench(fn, items):
        times = []
        for it in items:
            t0 = time.perf_counter()
            fn(it)
            times.append((time.perf_counter() - t0) * 1000)
        a = np.array(times)
        return {"n": len(a), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                "p95_ms": float(np.percentile(a, 95))}

    X = dataset.X[idx]
    sm, db = GestureSmoother(), Debouncer()
    t = [0.0]

    def smooth_step(tok):
        t[0] += 1 / 30
        db.update(sm.update(tok, t[0]), t[0])

    return {
        "machine": f"{sys.platform}, Python {sys.version.split()[0]}",
        "preprocess": bench(lambda r: preprocess(row_to_landmarks(*r)), raw_rows),
        "svm_predict_proba": bench(lambda x: model.predict_proba(x.reshape(1, -1)), X),
        "svm_predict": bench(lambda x: model.predict(x.reshape(1, -1)), X),
        "smoothing_plus_debounce": bench(smooth_step, list(dataset.y[idx])),
    }


def grid_facts(dataset, splits):
    """Re-run Siri's exact LOUO grid (train.py) to recover every combination's score."""
    train = np.isin(dataset.groups, splits.train_users)
    grid = {"svm__C": [0.1, 1, 10], "svm__kernel": ["linear", "rbf"]}  # copied from train.py
    search = GridSearchCV(build_pipeline(), grid, cv=LeaveOneGroupOut(), n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(dataset.X[train], dataset.y[train], groups=dataset.groups[train])
    res = search.cv_results_
    folds = [k for k in res if k.startswith("split") and k.endswith("_test_score")]
    rows = [{"C": float(res["param_svm__C"][i]), "kernel": str(res["param_svm__kernel"][i]),
             "mean": float(res["mean_test_score"][i]), "std": float(res["std_test_score"][i]),
             "folds": {f"held_out_{splits.train_users[j]}": float(res[folds[j]][i])
                       for j in range(len(folds))}}
            for i in range(len(res["params"]))]
    return {"rows": rows, "best_params": search.best_params_,
            "best_score": float(search.best_score_), "seconds": time.perf_counter() - t0}


def replay_facts(model, splits):
    """Replay one test-user session through the live classifier + smoother + debouncer.

    Rows are fed in file order at RECORD_SAMPLE_HZ (the recording rate), so this is an
    offline replay of real recorded landmarks, NOT a live measurement.
    """
    path = sorted((ROOT_DIR / "data" / "raw").glob(f"{splits.test_users[0]}_bright_near_*.csv"))[0]
    dataset, _ = load_dataset([path])
    classes = list(model.classes_)
    sm, db = GestureSmoother(), Debouncer()
    seq = []
    for i, (x, true) in enumerate(zip(dataset.X, dataset.y)):
        t = i / RECORD_SAMPLE_HZ
        p = model.predict_proba(x.reshape(1, -1))[0]
        lab, conf = classes[int(p.argmax())], float(p.max())
        raw = LABEL_NONE if conf < CONFIDENCE_THRESHOLD else lab
        stable = sm.update(raw, t)
        event = db.update(stable, t)
        seq.append({"t": t, "true": str(true), "raw": raw, "conf": conf,
                    "stable": stable, "event": event})
    return {"file": path.name, "hz": RECORD_SAMPLE_HZ, "sequence": seq,
            "n_events": sum(1 for s in seq if s["event"])}


def test_facts():
    xml = OUT_DIR / "pytest.xml"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"],
                          cwd=ROOT_DIR, capture_output=True, text=True)
    suite = ET.parse(xml).getroot().find("testsuite")
    per_file = {}
    for case in suite.iter("testcase"):
        f = case.get("classname").split(".")[1] + ".py"
        d = per_file.setdefault(f, {"passed": 0, "failed": 0, "skipped": 0})
        if case.find("failure") is not None or case.find("error") is not None:
            d["failed"] += 1
        elif case.find("skipped") is not None:
            d["skipped"] += 1
        else:
            d["passed"] += 1
    return {"summary_line": proc.stdout.strip().splitlines()[-1],
            "tests": int(suite.get("tests")), "failures": int(suite.get("failures")),
            "errors": int(suite.get("errors")), "skipped": int(suite.get("skipped")),
            "per_file": per_file}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits(SPLITS_PATH)
    dataset, cleaning = load_dataset(excluded_files=splits.excluded_files)
    model = joblib.load(MODEL_PATH)
    p = model.get_params()
    facts = {
        "model": {"path": str(MODEL_PATH.relative_to(ROOT_DIR)),
                  "size_bytes": os.path.getsize(MODEL_PATH),
                  "pipeline": [name for name, _ in model.steps],
                  "C": p["svm__C"], "kernel": p["svm__kernel"], "gamma": p["svm__gamma"],
                  "probability": p["svm__probability"],
                  "n_support_vectors": int(model.named_steps["svm"].n_support_.sum()),
                  "classes": list(map(str, model.classes_))},
        "dataset": dataset_facts(dataset, cleaning, splits),
        "test": model_test_facts(model, dataset, splits),
        "timing": timing_facts(model, dataset),
        "replay": replay_facts(model, splits),
        "pytest": test_facts(),
    }
    if not args.no_grid:
        facts["grid"] = grid_facts(dataset, splits)
    elif (OUT_DIR / "facts.json").exists():
        old = json.loads((OUT_DIR / "facts.json").read_text())
        if "grid" in old:
            facts["grid"] = old["grid"]
    (OUT_DIR / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
    short = {k: v for k, v in facts.items() if k != "replay"}
    short["replay"] = {k: v for k, v in facts["replay"].items() if k != "sequence"}
    print(json.dumps(short, indent=1, default=str))


if __name__ == "__main__":
    main()
