# Strata — notes for Claude Code

Affordable precision-ag data/ML product for small Pennsylvania farms: per-crop
yield regression, a soil-order classifier, a cited Penn State rule engine, a
probe calibration seam, and a FastAPI app over them. See `README.md` for setup
and `docs/TECHNICAL_REPORT.md` for how everything fits together.

- Run tests: `python -m pytest` (from the repo root; `pytest.ini` puts `src` on the path).
- Module entry points need `src` on the path: `PYTHONPATH=src python -m strata.main`.
- Rebuild served models after the analysis table changes: `python scripts/build_models.py`.

## Documentation policy
- docs/TECHNICAL_REPORT.md is the canonical technical writeup. Keep it accurate to the code.
- After any prompt that changes code, append an entry to docs/CHANGELOG.md:
  date, the prompt (see .claude/last_prompt.txt), files changed, what and why,
  and metric before/after if applicable.
- If architecture, data flow, or a module's behavior changed, update the relevant
  report section and any Mermaid diagram. Regenerate figures with
  `python docs/scripts/make_figures.py`.
- Diagrams: Mermaid in markdown. Charts: scripts in docs/scripts, output to docs/figures.
- Don't document from memory: read the file before describing it.
- Evaluation runs (`scripts/build_models.py`, `strata.main`) append a line to
  docs/metrics.jsonl automatically via `strata.metrics_log`. Link the changelog
  entry to the metrics line (by timestamp or commit) when a change moves a metric.
- The Stop hook (`.claude/hooks/check_docs.py`) blocks finishing a turn that
  changed code without touching docs/CHANGELOG.md.
