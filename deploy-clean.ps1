# PowerShell script to clean up for Vercel deployment

Write-Host "🧹 Cleaning up for Vercel deployment..." -ForegroundColor Green

# Remove Python cache
Write-Host "Removing Python cache..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Directory -Name "__pycache__" | ForEach-Object { Remove-Item -Path $_ -Recurse -Force -ErrorAction SilentlyContinue }
Get-ChildItem -Path . -Recurse -File -Name "*.pyc" | ForEach-Object { Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue }

# Remove model cache directories
Write-Host "Removing model cache directories..." -ForegroundColor Yellow
$cacheDirs = @(".cache", "models", "model_cache", "embeddings", "vector_cache", "chroma_db")
foreach ($dir in $cacheDirs) {
    if (Test-Path $dir) {
        Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Removed $dir" -ForegroundColor Gray
    }
}

# Remove any large model files
Write-Host "Removing large model files..." -ForegroundColor Yellow
$modelExtensions = @("*.bin", "*.safetensors", "*.ckpt", "*.pth", "*.pt", "*.h5", "*.hdf5", "*.onnx", "*.tflite", "*.pb", "*.pkl", "*.joblib")
foreach ($ext in $modelExtensions) {
    Get-ChildItem -Path . -Recurse -File -Name $ext | ForEach-Object { 
        Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue 
        Write-Host "Removed $_" -ForegroundColor Gray
    }
}

# Remove virtual environment if exists
Write-Host "Removing virtual environment..." -ForegroundColor Yellow
$venvDirs = @("venv", "env", ".venv")
foreach ($dir in $venvDirs) {
    if (Test-Path $dir) {
        Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Removed $dir" -ForegroundColor Gray
    }
}

# Remove any database files
Write-Host "Removing database files..." -ForegroundColor Yellow
$dbExtensions = @("*.db", "*.sqlite", "*.sqlite3")
foreach ($ext in $dbExtensions) {
    Get-ChildItem -Path . -Recurse -File -Name $ext | ForEach-Object { 
        Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue 
        Write-Host "Removed $_" -ForegroundColor Gray
    }
}

Write-Host "✅ Cleanup complete! Ready for Vercel deployment." -ForegroundColor Green
Write-Host "📦 Your deployment bundle should now be much smaller." -ForegroundColor Cyan
