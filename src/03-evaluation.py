import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from config import DATA_DIR, LABEL_MAP
from utils import get_logger

# Importáljuk a modellt a training fájlból
# Megjegyzés: Dockerben a munkakönyvtár /app, így a src.02-training import működik
# Vagy dinamikus import, ha a fájlnév '02-training.py'
import sys
sys.path.append(os.path.dirname(__file__))
# Mivel a fájlnév kötőjeles (02-training), importlibet használunk
import importlib
training_module = importlib.import_module("02-training")
BullFlagCNN = training_module.BullFlagCNN

logger = get_logger("EVALUATION")

def evaluate_model():
    logger.info("Starting evaluation on TEST set...")
    
    # 1. Adatok és Modell betöltése
    test_path = os.path.join(DATA_DIR, 'test_dataset.npz')
    model_path = os.path.join(DATA_DIR, 'bullflag_model.pth')

    if not os.path.exists(test_path) or not os.path.exists(model_path):
        logger.error("Test data or Model file not found. Run training first.")
        return

    # Adatok
    data = np.load(test_path)
    X_test = torch.tensor(data['X'], dtype=torch.float32)
    y_test = torch.tensor(data['y'], dtype=torch.long)
    
    # Modell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(LABEL_MAP)
    model = BullFlagCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logger.info(f"Loaded model and test set with {len(y_test)} samples.")

    # 2. Inference
    all_preds = []
    all_labels = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i+batch_size].to(device)
            labels = y_test[i:i+batch_size]
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Metrikák
    # Visszafejtjük az ID-kat szöveges címkékre a report kedvéért
    id_to_label = {v: k for k, v in LABEL_MAP.items()}
    target_names = [id_to_label[i] for i in range(num_classes) if i in np.unique(all_labels)]
    
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    logger.info("\n" + report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Összesített pontosság
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    logger.info(f"Final Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    evaluate_model()