param(
  [string]$VenvPath = ".venv"
)

python -m venv $VenvPath
& "$VenvPath\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
