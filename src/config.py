import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data') # Mounted in Docker
RAW_CSV_DIR = os.path.join(DATA_DIR, 'raw_csvs') # User must place CSVs here
LABEL_FILE = os.path.join(DATA_DIR, 'project-labels.json') # User must place JSON here
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_dataset.npz')
LOG_DIR = os.path.join(BASE_DIR, 'log')
LOG_FILE = os.path.join(LOG_DIR, 'run.log')

# Model Hyperparameters
# Based on common defaults for time-series deep learning
BATCH_SIZE = 16
EPOCHS = 20 # Can be increased for better performance
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 60 # Number of time steps to feed the model

# Labels Mapping (Based on PDF Page 4 XML) [cite: 82]
LABEL_MAP = {
    "Background": 0, # No pattern
    "Bullish Normal": 1,
    "Bullish Wedge": 2,
    "Bullish Pennant": 3,
    "Bearish Normal": 4,
    "Bearish Wedge": 5,
    "Bearish Pennant": 6
}
import os
