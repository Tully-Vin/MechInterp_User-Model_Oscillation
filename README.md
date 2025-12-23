# High-Gain User-Model Oscillations

This workspace supports a small, time-boxed study of oscillations in conditional LLM generation when the model infers user expertise and receives mixed feedback.

Quick start
1) Install torch with CUDA (see docs/INSTALL.md)
2) Run scripts/setup.ps1
3) Run scripts/run_generation.ps1
4) Run scripts/analyze.ps1
5) Optional: run scripts/run_probe.ps1

Project layout
- configs/ : experiment configs
- prompts/ : prompt library and jargon lists
- src/ : generation, analysis, and probe code
- docs/ : plan, methods, metrics, write-up outline
- data/ : raw outputs and probe artifacts
- results/ : figures and tables
- notes/ : worklog and open questions
