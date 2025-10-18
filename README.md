# Deep Learning-Based Sleep Stage Classification System

This project is a web application built with Flask that allows users to visualize and classify sleep stages from biological signal files (EDF). The system utilizes two advanced Deep Learning models, **Sleep Transformer** and **DeepSleepNet**, to predict sleep states (Wake, N1, N2, N3, REM) from three primary signal channels: EEG, EOG, and EMG.

## Table of Contents
- [Key Features](#key-features)
- [Workflow](#workflow)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Data Requirements](#data-requirements)
- [Setup & Installation Guide](#setup--installation-guide)
- [Technology Stack](#technology-stack)

## Key Features

- **Multi-Model Classification**: Integrates two powerful models (`MultiChannelSleepTransformer` and `MultiChannelDeepSleepNet`) for classification and result comparison.
- **Signal Visualization**: Uses Chart.js to plot EEG, EOG, and EMG signals, allowing users to zoom and pan for detailed analysis.
- **Flexible Interaction**:
    - Provides a list of pre-loaded sample EDF files for quick analysis.
    - Allows users to upload their own 30-second EDF files for instant prediction.
- **Visual Comparison**: Automatically compares the AI's prediction results with the ground-truth label extracted from the filename, highlighting accuracy.
- **User-Friendly Interface**: A modern, easy-to-use web interface built with Bootstrap.

## Workflow

1.  **File Selection/Upload**: The user can either select an existing EDF file from the `datasets` directory or upload a new file through the web interface.
2.  **Backend Processing (Flask)**:
    - `app.py` receives the request.
    - The `mne` library is used to read and extract data from the three signal channels (`EEG Fpz-Cz`, `EOG horizontal`, `EMG submental`) from the EDF file.
    - The data is preprocessed using Z-score normalization, mirroring the process used during model training.
    - The normalized data is fed into the two pre-trained models (`model_transformer` and `model_deepsleepnet`).
3.  **Prediction**: Both models output predictions for the five sleep stages. The backend calculates the probability distribution for each stage.
4.  **Return Results to Frontend**:
    - The backend sends a JSON response containing:
        - The prediction results and probability distributions from both models.
        - The ground-truth label extracted from the filename.
        - The raw signal data for all three channels to be plotted.
5.  **Display on Interface**:
    - Client-side JavaScript dynamically updates the UI with the received data.
    - Result cards are displayed for each model, comparing the prediction against the ground truth.
    - Chart.js is used to render the 3-channel signal plot.

## Model Architecture

The project employs two distinct architectures to process time-series data from biological signals, both adapted to handle 3-channel inputs.

1.  **MultiChannelSleepTransformer (`model.py`)**:
    - **Concept**: Leverages the Transformer architecture, highly successful in Natural Language Processing, to capture long-range dependencies within the sleep signals.
    - **Structure**:
        - Each channel (EEG, EOG, EMG) is processed by a separate Transformer branch.
        - Within each branch, the signal is divided into "patches," which are embedded and combined with Positional Encoding.
        - These embeddings are then passed through a Transformer Encoder.
        - The output features from the three branches are concatenated and passed through a final classifier.

2.  **MultiChannelDeepSleepNet (`model.py`)**:
    - **Concept**: Based on the well-known DeepSleepNet architecture, which combines Convolutional Neural Networks (CNNs) for local feature extraction and Recurrent Neural Networks (LSTMs) for learning sequential relationships.
    - **Structure**:
        - Similar to the Transformer model, each channel is processed by its own DeepSleepNet branch.
        - Each branch consists of two CNN streams with different kernel sizes to learn features at various frequencies.
        - The output of the CNN layers is fed into a Bidirectional LSTM network.
        - The final features from the three branches are concatenated and classified.

## Project Structure

```
.
├── app.py                      # Main Flask application file
├── generate_demo.py            # Script to generate synthetic EDF data for demo purposes
├── model.py                    # PyTorch model architecture definitions
├── requirements.txt            # List of required Python libraries
├── train.ipynb                 # (Optional) Jupyter Notebook for the training process
├── datasets/                   # Directory containing sample EDF files
│   ├── SC4032E_epoch_000_label_W.edf
│   └── ...
├── model_weights/              # Directory for pre-trained model weights
│   ├── MultiChannelDeepSleepNet_best.pt
│   └── MultiChannelSleepTransformer_best.pt
├── static/
│   └── css/
│       └── style.css           # Custom CSS file
└── templates/
    ├── index.html              # Main HTML template for the UI
    └── ecg.html                # (Optional) Template for another feature
```

## Data Requirements

- **Format**: The system requires input files in **EDF (European Data Format)**.
- **File Structure**: Each EDF file must be a 30-second *epoch* containing at least the following three channels:
    - `EEG Fpz-Cz`
    - `EOG horizontal`
    - `EMG submental`
- **Sampling Frequency**: The models were trained on data with a sampling frequency of **100 Hz**.
- **Sample Data Generation**: You can run the `generate_demo.py` script to create a demo dataset with simulated signal characteristics for each sleep stage.
  ```bash
  python generate_demo.py
  ```
  This command will create a `generated_full_demo_dataset` directory containing 5 sample EDF files.

## Setup & Installation Guide

**Prerequisites**: Python 3.8+ and pip.

**Step 1: Clone the repository**
```bash
git clone https://your-repository-link.git
cd your-repository-directory
```

**Step 2: Create a virtual environment (recommended)**
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

**Step 3: Install the required libraries**
```bash
pip install -r requirements.txt
```

**Step 4: Prepare data and model weights**
- Ensure you have the `datasets` directory populated with sample EDF files.
- Ensure you have the `model_weights` directory containing the pre-trained `.pt` weight files.

**Step 5: Run the Flask application**
```bash
python app.py
```

**Step 6: Access the application**
Open your web browser and navigate to: `http://127.0.0.1:5000`

## Technology Stack

- **Backend**: Flask, PyTorch, MNE-Python, NumPy
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript, Chart.js, chartjs-plugin-zoom
- **AI Models**: Transformer, CNN, LSTM
