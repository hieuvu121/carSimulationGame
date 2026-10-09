
# <Name> – <role, e.g. Hand tracking> — Technical notes
Author: <name>   |   Files owned: <list>   |   Requirement IDs: <e.g. HT.1–HT.4, R3, R4>

## 1. What it does (3–5 sentences)
Plain-language description + where it sits in the pipeline (link to DESIGN.md §2).

## 2. Design decisions
| Decision | Alternatives considered | Why we chose it |
|---|---|---|
| e.g. palm-length scaling | max-distance scaling, bounding-box scaling | gesture-independent → fist and palm get comparable scales |

## 3. Interfaces provided / consumed
Signatures (copied from interfaces.md §4) and any deviation, with justification.

## 4. Challenges and how we solved them
- Challenge → what we tried → what worked (with evidence: a test, a number, a plot).

## 5. Results
Key numbers/figures from results/ relevant to this part (e.g. test count & pass rate,
accuracy, latency, FPS). Each figure: file path + one-sentence interpretation.
Mark which results are [Must] and which are [Nice].

## 6. Limitations & future work
Honest list (2–4 bullets).

## 7. Slide summary (max 40 words + 1 figure)
The exact text and figure to put on the slide.


## 8. MileStone
10/10/2026
Iteration 1:
Triet — webcam + MediaPipe verification passed on Windows using 64-bit Python 3.12.10, MediaPipe 0.10.21 and OpenCV 4.11.0. The check script successfully captured my hand and printed 21 landmarks. Output attached below.
Please check carSimulationGame/src/capture/check/check_hand.py

Triet — camera FPS check completed. Bright: 29.94 FPS; dim: 29.96 FPS. Each measurement ran for approximately 10 seconds after 30 warm-up frames. My camera maintained approximately 30 FPS in both conditions, so I did not observe the expected dim-light slowdown.
Please check carSimulationGame/src/capture/check/check_fps.py



```


