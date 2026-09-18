# Schedule Risk Analyzer

Upload three-point estimates, run a Monte Carlo, see where your deadline actually lands.

## What it does

Takes tasks with optimistic / most likely / pessimistic durations plus dependencies, samples each task from a PERT distribution 10,000 times, walks the DAG in topological order each run, and reports P50 / P70 / P80 / P90 / P95, the chance of hitting your target, and which tasks drive the variance. A Gemini call then turns those numbers into plain English plus mitigations.

## PERT methodology

Mean per task is `(O + 4M + P) / 6`. Sampling fits a Beta scaled to `[O, P]` with `alpha = 1 + 4(M-O)/(P-O)` and `beta = 1 + 4(P-M)/(P-O)`, so draws cluster around M instead of spreading flat like a triangular. Zero spread (`O == P`) is deterministic. Each iteration samples every task, forward pass is `start = max(pred finish)`, `finish = start + sample`, project completion is the max over leaf tasks, so parallel paths resolve by taking the longest. Risk contribution is the Pearson correlation between each task's samples and total completion across runs.

## Where AI is used (and where it is not)

`simulation_engine.py` is pure math, no AI, no Flask. `ai_interpreter.py` is the only module that calls Gemini. It sends P50/P80/P95, on-time chance, the P95-P50 gap, and the top 5 drivers with O/M/P and correlation, and asks for JSON back: a short summary, 3 task-specific mitigations tied to each spread, and a buffer for ~80% confidence. If the key is missing, the call times out, or the JSON does not parse, the page still renders everything else with an "AI interpretation unavailable" note. Prompt and raw response go to `logs/ai_debug.log` for the writeup.

## Run it

```
pip install -r requirements.txt
cp .env.example .env   # put your GEMINI_API_KEY in there
python app.py
```

Open http://127.0.0.1:5000, drop `sample_input.json` or paste JSON, set simulations (1000-50000), Run Simulation.

Tests: `python -m pytest tests/ -q`

## Known limitation

The simulation runs synchronously in the request thread. Fine locally (10k runs is a few seconds), but on production this should move to a background worker with polling.
