# Contributing to CAESAR

CAESAR is an experimental recurrent actor-critic research codebase. Reproducible negative results are as welcome as positive results.

## Development

Use Python 3.10 or newer and a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

For changes to the agent or environment, run a small seeded smoke test before proposing expensive training. Report the exact command, commit, dependency versions, hardware and actual test result.

## Research contributions

- State the hypothesis, endpoint and controls before examining held-out outcomes.
- Preserve the existing experiment version and configuration; new variants should be explicitly named.
- Record training and evaluation seeds separately, retain raw results, and compare against matched nonlearning controls.
- Distinguish unit tests, short diagnostic runs and independent replication. Do not infer intelligence or consciousness from internal state metrics alone.
- Cite prior experiments, including negative results. Start with the [current findings](README.md#current-research-findings) and the [credit-assignment investigation](https://github.com/workingclassbuddha/Caeser/issues/17).

Use the issue tracker for proposed experiments or reproducible bugs. Keep private data, credentials and unrelated runtime artifacts out of submissions.
