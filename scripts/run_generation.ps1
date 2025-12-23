param(
  [string]$Config = "configs/experiment.yaml"
)

& ".venv\Scripts\python.exe" src\run_generation.py --config $Config
