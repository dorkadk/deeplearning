# Deep Learning Class (VITMMA19) Project Work

## Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Deák Dorka
- **Aiming for +1 Mark**: No

### Solution Description

This project implements a Deep Learning solution to detect and classify "Bull flag" and "Bear flag" patterns in financial time-series data (OHLC).

**The Problem:** Financial markets exhibit specific consolidation patterns (flags) after strong trends (poles). Identifying these correctly (Normal, Wedge, Pennant) is crucial for algorithmic trading strategies.

**Model Architecture:** The solution uses a **1D Convolutional Neural Network (1D-CNN)**. 
* **Input:** A sequence of OHLC data (normalized percent change).
* **Layers:** Two convolutional layers with BatchNorm and ReLU activation to extract local temporal features, followed by MaxPool layers.
* **Classifier:** A fully connected dense layer maps the extracted features to one of the 7 classes (Background + 2 Directions * 3 Types).

**Methodology:** 1.  **Data Prep:** Raw CSVs are matched with Label Studio JSON annotations. Segments are extracted, padded to a fixed sequence length (60), and normalized.
2.  **Training:** The model is trained using CrossEntropyLoss and the Adam optimizer.
3.  **Logging:** All metrics (Loss, Accuracy) are logged to `log/run.log` for review.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
docker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1