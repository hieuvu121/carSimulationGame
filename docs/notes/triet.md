# Technical Notes Template

Part of the design docs — start at [DESIGN.md](../DESIGN.md) for the index.

---

## 14. Technical notes template

Each person copies this into `docs/notes/<name>.md` (`triet.md`, `uy.md`, `shobita.md`, `siri.md`, `evan.md`). The notes are written during Iteration 5 and reviewed by the Lead. They feed 1–2 slides per person.

```markdown
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
```
