#!/bin/bash
set -e

# Print environment info
echo "Starting PDF Heading Extraction Pipeline..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Available files in /app/input:"
ls -la /app/input/ || echo "Input directory not found or empty"
echo "Available models:"
ls -la /app/models/ || echo "Models directory not found"

# Ensure output directory exists
mkdir -p /app/output

# Run the main pipeline
echo "Running heading extraction pipeline..."
cd /app
python -m src.main_pipeline

echo "Pipeline execution completed!"
echo "Output files generated:"
ls -la /app/output/ || echo "No output files generated"