# Install and setup

Recommended
- GPU: RTX 3060 Ti
- Python: 3.10+ (3.12 works)

1) Install torch with CUDA
- Example for CUDA 12.1
  pip install torch --index-url https://download.pytorch.org/whl/cu121

2) Create venv and install deps
- Run:
  scripts\setup.ps1

3) Verify
- Run:
  python -c "import torch; print(torch.cuda.is_available())"

4) First run
- scripts\run_generation.ps1
- scripts\analyze.ps1
- optional: scripts\run_probe.ps1
