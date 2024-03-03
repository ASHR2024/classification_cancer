#!/bin/bash
echo "Starting mlp pipeline"

# Change directory and run file
cd src/
python app_cancer.py

# Pause
echo "Program completed"
/bin/bash
read 