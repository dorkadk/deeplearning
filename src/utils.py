import logging
import sys
import os
from . import config

def get_logger(name):
    # Ensure log directory exists
    os.makedirs(config.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers are already added to avoid duplicates
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout) # For Docker logs
        f_handler = logging.FileHandler(config.LOG_FILE, mode='w') # For grading submission

        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger