param(
  [string]$Input = ""
)

if ($Input -eq "") {
  & ".venv\Scripts\python.exe" src\analyze_results.py
} else {
  & ".venv\Scripts\python.exe" src\analyze_results.py --input $Input
}
