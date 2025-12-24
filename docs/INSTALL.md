# Install and setup

Recommended
- GPU: RTX 3060 Ti
- Python: 3.10+ (3.12 works)

1) Install torch with CUDA
- Example for CUDA 12.1
  pip install torch --index-url https://download.pytorch.org/whl/cu121

2) Create venv and install deps
- PowerShell (recommended):
  .\scripts\setup.ps1
- CMD:
  scripts\setup.cmd
Note: do not run .ps1 files with Python. Use PowerShell or the .cmd wrappers.

3) Verify
- Run:
  python -c "import torch; print(torch.cuda.is_available())"
- If False, reinstall torch with CUDA and check that `nvidia-smi` works.

4) First run
- PowerShell:
  .\scripts\run_generation.ps1
  .\scripts\analyze.ps1
  .\scripts\run_probe.ps1
- CMD:
  scripts\run_generation.cmd
  scripts\analyze.cmd
  scripts\run_probe.cmd
