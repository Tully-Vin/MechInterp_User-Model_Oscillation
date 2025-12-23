param(
  [string]$Data = ""
)

if ($Data -eq "") {
  & ".venv\Scripts\python.exe" src\probe.py
} else {
  & ".venv\Scripts\python.exe" src\probe.py --data $Data
}
