#!/bin/sh

LOGFILE="log/run.log"
mkdir -p log

echo "=== DATA PREPROCESSING ===" >> "$LOGFILE" 2>&1
python src/01-data-preprocessing.py >> "$LOGFILE" 2>&1

echo "=== TRAINING ===" >> "$LOGFILE" 2>&1
python src/02-training.py >> "$LOGFILE" 2>&1

echo "=== EVALUATION ===" >> "$LOGFILE" 2>&1
python src/03-evaluation.py >> "$LOGFILE" 2>&1
