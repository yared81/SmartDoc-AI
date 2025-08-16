#!/bin/bash

echo "🧹 Cleaning up for Vercel deployment..."

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove model cache directories
echo "Removing model cache directories..."
rm -rf .cache/ 2>/dev/null || true
rm -rf models/ 2>/dev/null || true
rm -rf model_cache/ 2>/dev/null || true
rm -rf embeddings/ 2>/dev/null || true
rm -rf vector_cache/ 2>/dev/null || true
rm -rf chroma_db/ 2>/dev/null || true

# Remove any large model files
echo "Removing large model files..."
find . -name "*.bin" -delete 2>/dev/null || true
find . -name "*.safetensors" -delete 2>/dev/null || true
find . -name "*.ckpt" -delete 2>/dev/null || true
find . -name "*.pth" -delete 2>/dev/null || true
find . -name "*.pt" -delete 2>/dev/null || true
find . -name "*.h5" -delete 2>/dev/null || true
find . -name "*.hdf5" -delete 2>/dev/null || true
find . -name "*.onnx" -delete 2>/dev/null || true
find . -name "*.tflite" -delete 2>/dev/null || true
find . -name "*.pb" -delete 2>/dev/null || true
find . -name "*.pkl" -delete 2>/dev/null || true
find . -name "*.joblib" -delete 2>/dev/null || true

# Remove virtual environment if exists
echo "Removing virtual environment..."
rm -rf venv/ 2>/dev/null || true
rm -rf env/ 2>/dev/null || true
rm -rf .venv/ 2>/dev/null || true

# Remove any database files
echo "Removing database files..."
find . -name "*.db" -delete 2>/dev/null || true
find . -name "*.sqlite" -delete 2>/dev/null || true
find . -name "*.sqlite3" -delete 2>/dev/null || true

echo "✅ Cleanup complete! Ready for Vercel deployment."
echo "📦 Your deployment bundle should now be much smaller."
