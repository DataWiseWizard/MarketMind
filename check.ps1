# MarketMind Windows PowerShell Quick Check Script
# Usage: .\check.ps1

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  MarketMind - Lint & Test Suite (Windows)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$VENV_PYTHON = ".\.venv\Scripts\python.exe"

if (-Not (Test-Path $VENV_PYTHON)) {
    $VENV_PYTHON = "python"
}

Write-Host "`n[1/2] Running Ruff Linter..." -ForegroundColor Yellow
& $VENV_PYTHON -m ruff check src/ tests/ dashboard/
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nRuff linter found issues. Run 'python -m ruff check --fix src/ tests/ dashboard/' to auto-fix." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[OK] Ruff check passed with 0 errors!" -ForegroundColor Green

Write-Host "`n[2/2] Running Pytest Suite..." -ForegroundColor Yellow
& $VENV_PYTHON -m pytest tests/ -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPytest suite failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[SUCCESS] All checks passed! Safe to commit and push." -ForegroundColor Green
