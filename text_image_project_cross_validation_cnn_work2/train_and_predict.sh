#!/bin/bash

# Run the training script
echo "Starting training..."
python3 train.py

# Once training is done, run the prediction script
echo "Training completed. Running predictions..."
python3 predict.py
