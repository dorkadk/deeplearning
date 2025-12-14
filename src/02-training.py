import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os
from config import PROCESSED_DATA_PATH, DATA_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, LABEL_MAP
from utils import get_logger

logger = get_logger("TRAINING")

# --- MODELL DEFINÍCIÓ ---
# Ezt a osztályt importálja majd a 03 és 04 script is
class BullFlagCNN(nn.Module):
    def __init__(self, num_classes):
        super(BullFlagCNN, self).__init__()
        # Input: (Batch, 4 features, 60 sequence_length)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.flatten = nn.Flatten()
        
        # A 60-as szekvencia 3 pool réteg után kb 7-es hosszúságú lesz (60->30->15->7)
        # 128 csatorna * 7 hossz = 896 input a Dense rétegnek
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # PyTorch Conv1d elvárása: (Batch, Channels, Length)
        # Az adatunk: (Batch, Length, Channels) -> (Batch, 60, 4)
        # Permutálni kell:
        x = x.permute(0, 2, 1) 
        
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def train_model():
    logger.info("=== Configuration ===")
    logger.info(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # 1. Adat betöltése
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error(f"Processed data missing: {PROCESSED_DATA_PATH}")
        return

    data = np.load(PROCESSED_DATA_PATH)
    X_all = torch.tensor(data['X'], dtype=torch.float32)
    y_all = torch.tensor(data['y'], dtype=torch.long)

    # 2. Split: Train (70%), Validation (15%), Test (15%)
    total_count = len(X_all)
    train_count = int(0.7 * total_count)
    val_count = int(0.15 * total_count)
    test_count = total_count - train_count - val_count

    dataset = TensorDataset(X_all, y_all)
    train_set, val_set, test_set = random_split(dataset, [train_count, val_count, test_count], generator=torch.Generator().manual_seed(42))

    logger.info(f"Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # Mentsük el a TEST setet külön, hogy a 03-evaluation.py tiszta adatokon dolgozhasson
    # Kinyerjük az indexeket és elmentjük numpy formátumban
    test_indices = test_set.indices
    X_test = X_all[test_indices].numpy()
    y_test = y_all[test_indices].numpy()
    test_path = os.path.join(DATA_DIR, 'test_dataset.npz')
    np.savez(test_path, X=X_test, y=y_test)
    logger.info(f"Test dataset saved to {test_path} for independent evaluation.")

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # 3. Modell inicializálása
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model = BullFlagCNN(num_classes=len(LABEL_MAP)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'bullflag_model.pth'))

    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.1f}%")

if __name__ == "__main__":
    train_model()