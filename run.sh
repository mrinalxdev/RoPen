#!/bin/bash

# Tiny Shakespeare Pure Attention Transformer
# Usage: ./run.sh [train|sample|both]

set -e  # Exit on error

MODE=${1:-both}

echo "Tiny Shakespeare Pure Attention Transformer"
echo "================================================"

mkdir -p logs

case $MODE in
    "train")
        echo "Starting training..."
        python train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    "sample")
        echo "Generating samples..."
        python sample.py 2>&1 | tee logs/sample_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    "both")
        echo "Starting training..."
        python train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
        
        echo ""
        echo "Generating samples..."
        python sample.py 2>&1 | tee logs/sample_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: ./run.sh [train|sample|both]"
        exit 1
        ;;
esac

echo ""
echo "Done!"