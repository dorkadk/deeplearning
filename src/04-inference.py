import torch
import pandas as pd
import numpy as np
import os
import importlib
from config import RAW_CSV_DIR, DATA_DIR, LABEL_MAP, SEQUENCE_LENGTH
from utils import get_logger

# Modell importálása
training_module = importlib.import_module("02-training")
BullFlagCNN = training_module.BullFlagCNN

logger = get_logger("INFERENCE")

def run_inference():
    logger.info("Starting inference on raw CSV files...")

    # Keressünk egy CSV fájlt a raw könyvtárban
    if not os.path.exists(RAW_CSV_DIR):
        logger.error(f"Raw CSV directory not found: {RAW_CSV_DIR}")
        return

    files = [f for f in os.listdir(RAW_CSV_DIR) if f.endswith('.csv')]
    if not files:
        logger.error("No CSV files found for inference.")
        return

    # Vegyük az elsőt példának
    target_file = files[0]
    csv_path = os.path.join(RAW_CSV_DIR, target_file)
    logger.info(f"Processing file: {target_file}")

    # 1. Adat betöltése és előkészítése (ugyanaz mint prep-nél)
    df = pd.read_csv(csv_path)
    
    # Timestamp kezelés
    if 'timestamp' in df.columns:
        if df['timestamp'].dtype == 'int64':
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature engineering (pct_change)
    cols = ['open', 'high', 'low', 'close']
    features = df[cols].pct_change().fillna(0).values
    timestamps = df['timestamp'].values

    # 2. Sliding Window létrehozása
    # Végigmegyünk az idősoron és 60-as ablakokat vágunk ki
    windows = []
    window_timestamps = []
    
    stride = 10  # Nem minden egyes lépésben lépünk, hanem 10-esével a gyorsaság miatt
    
    for i in range(0, len(features) - SEQUENCE_LENGTH, stride):
        window = features[i : i + SEQUENCE_LENGTH]
        windows.append(window)
        # A minta idejének a végét rögzítjük
        window_timestamps.append(timestamps[i + SEQUENCE_LENGTH - 1])

    if not windows:
        logger.warning("File too short for inference.")
        return

    X_infer = np.array(windows)
    X_tensor = torch.tensor(X_infer, dtype=torch.float32)

    # 3. Modell futtatása
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BullFlagCNN(num_classes=len(LABEL_MAP)).to(device)
    
    model_path = os.path.join(DATA_DIR, 'bullflag_model.pth')
    if not os.path.exists(model_path):
        logger.error("Model not found.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logger.info(f"Running prediction on {len(X_tensor)} windows...")
    
    predictions = []
    probs = []
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            
            # Softmax a valószínűségekhez
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, 1)
            
            predictions.extend(preds.cpu().numpy())
            probs.extend(max_probs.cpu().numpy())

    # 4. Eredmények kiértékelése
    id_to_label = {v: k for k, v in LABEL_MAP.items()}
    found_patterns = 0

    print("\n--- DETECTION REPORT ---")
    for i, pred_id in enumerate(predictions):
        label = id_to_label[pred_id]
        confidence = probs[i]
        
        # Csak akkor jelezzük, ha NEM Background és a bizonyosság elég nagy (>80%)
        if label != "Background" and confidence > 0.80:
            ts = window_timestamps[i]
            print(f"[{ts}] Detected {label} (Conf: {confidence:.2f})")
            found_patterns += 1

    if found_patterns == 0:
        logger.info("No significant patterns detected in this file.")
    else:
        logger.info(f"Total patterns detected: {found_patterns}")

if __name__ == "__main__":
    run_inference()