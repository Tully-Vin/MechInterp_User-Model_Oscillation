# High-Gain User-Model Oscillations

This workspace supports a small, time-boxed study of oscillations in conditional LLM generation when the model infers user expertise and receives mixed feedback.

Quick start
1) Install torch with CUDA (see docs/INSTALL.md)
2) PowerShell: .\scripts\setup.ps1 (or CMD: scripts\setup.cmd)
3) PowerShell: .\scripts\run_generation.ps1 (or CMD: scripts\run_generation.cmd)
4) PowerShell: .\scripts\analyze.ps1 (or CMD: scripts\analyze.cmd)
5) Optional: .\scripts\run_probe.ps1 (or scripts\run_probe.cmd)
6) Explorer UI: .\scripts\run_explorer.ps1 (or scripts\run_explorer.cmd)

Project layout
- configs/ : experiment configs
- prompts/ : prompt library and jargon lists
- src/ : generation, analysis, and probe code
- docs/ : plan, methods, metrics, write-up outline
- data/ : raw outputs and probe artifacts
- results/ : figures and tables
- notes/ : worklog and open questions
