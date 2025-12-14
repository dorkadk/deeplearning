import pandas as pd
import numpy as np
import json
import os
from config import RAW_CSV_DIR, LABEL_FILE, PROCESSED_DATA_PATH, LABEL_MAP, SEQUENCE_LENGTH
from utils import get_logger

logger = get_logger("PREPROCESSING")

def load_and_process_data():
    logger.info("Starting data preprocessing...")
    
    if not os.path.exists(LABEL_FILE):
        logger.error(f"Label file not found at {LABEL_FILE}. Make sure to mount it or put it in data/.")
        return

    with open(LABEL_FILE, 'r') as f:
        labels_json = json.load(f)

    X = []
    y = []
    processed_files_count = 0

    # Iterate through each task in the JSON
    for entry in labels_json:
        # 1. Resolve Filename
        # Label Studio often prepends a UUID (e.g., "d992bfe2-aapl_5min.csv")
        # We need to find the matching file in raw_csvs directory.
        raw_filename = entry['file_upload']
        
        # Try finding exact match
        csv_path = os.path.join(RAW_CSV_DIR, raw_filename)
        
        # If not found, try stripping the UUID (everything before the first hyphen)
        if not os.path.exists(csv_path) and '-' in raw_filename:
            clean_name = raw_filename.split('-', 1)[1] # "uuid-name.csv" -> "name.csv"
            csv_path = os.path.join(RAW_CSV_DIR, clean_name)
            
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file for '{raw_filename}' not found in {RAW_CSV_DIR}. Skipping.")
            continue

        processed_files_count += 1
        
        # 2. Load CSV Data
        try:
            df = pd.read_csv(csv_path)
            
            # Standardize column names to lowercase just in case
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure timestamp column exists and is datetime
            # Your JSON uses "2025-10-01 08:45", so we ensure the CSV matches that
            if 'timestamp' in df.columns:
                # Attempt to parse timestamp. Handles unix ms, unix s, or string formats.
                if df['timestamp'].dtype == 'int64':
                     # If unix timestamp in ms
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                logger.error(f"Column 'timestamp' missing in {csv_path}")
                continue

            # [cite_start]Calculate Features (Percent Change) [cite: 38]
            # We assume columns: open, high, low, close exist
            features = df[['open', 'high', 'low', 'close']].pct_change().fillna(0).values
            timestamps = df['timestamp'].values

            # 3. Parse Annotations
            # Your JSON structure: entry -> annotations -> result -> value -> timeserieslabels
            for annotation in entry.get('annotations', []):
                for result in annotation.get('result', []):
                    val = result.get('value', {})
                    
                    if 'timeserieslabels' in val:
                        label_name = val['timeserieslabels'][0]
                        
                        # Parse start/end from JSON
                        start_time = pd.to_datetime(val['start'])
                        end_time = pd.to_datetime(val['end'])

                        # Create mask to find rows within this timeframe
                        mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                        selected_data = features[mask]

                        if len(selected_data) > 0:
                            # Pad or truncate to SEQUENCE_LENGTH
                            if len(selected_data) >= SEQUENCE_LENGTH:
                                processed_seq = selected_data[:SEQUENCE_LENGTH]
                            else:
                                # Zero padding if too short
                                padding = np.zeros((SEQUENCE_LENGTH - len(selected_data), 4))
                                processed_seq = np.vstack((selected_data, padding))

                            label_id = LABEL_MAP.get(label_name, 0)
                            
                            X.append(processed_seq)
                            y.append(label_id)
                        else:
                            logger.debug(f"Label {label_name} found but no matching data points in {csv_path} between {start_time} and {end_time}")

        except Exception as e:
            logger.error(f"Error processing {csv_path}: {e}")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        logger.error("No valid samples were extracted! Check your CSV timestamps vs JSON timestamps.")
    else:
        logger.info(f"Data processing complete. Processed {processed_files_count} files.")
        logger.info(f"Dataset Shape: {X.shape}, Labels: {y.shape}")
        
        # Save
        np.savez(PROCESSED_DATA_PATH, X=X, y=y)
        logger.info(f"Saved processed data to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    load_and_process_data()