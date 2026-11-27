# MarketMind Docker Runner Script
# Usage: .\run_docker.ps1

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  MarketMind - Full Stack Docker Orchestration" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

Write-Host "`nBuilding and launching containers (FastAPI, Streamlit, MLflow)..." -ForegroundColor Yellow
docker-compose up --build -d

Write-Host "`nWaiting for services to become healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

docker-compose ps

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "  MarketMind Services Active:" -ForegroundColor Green
Write-Host "  - Streamlit Dashboard : http://localhost:8501" -ForegroundColor Green
Write-Host "  - FastAPI Swagger Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  - MLflow Tracking UI  : http://localhost:5000" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "`nTo inspect logs: docker-compose logs -f" -ForegroundColor Cyan
Write-Host "To stop containers: docker-compose down" -ForegroundColor Cyan
