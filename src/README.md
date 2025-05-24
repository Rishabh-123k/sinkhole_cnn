Sinkhole CNN

A Python project for detecting sinkhole-like depressions in 1D elevation profiles, extracted from Hector SLAM maps using a 1D CNN.

Project Structure

sinkhole_cnn/
├── data/              # Synthetic elevation profiles (auto-generated)
├── models/            # Trained model weights
├── notebooks/         # Jupyter notebooks stuff
├── src/               # Source code
│   ├── generate_data.py  # Synthetic data generator for sinkhole and non-sinkhole profiles
│   ├── model.py          # 1D CNN model definition
│   ├── train.py          # Training script (loads data, trains CNN, saves weights)
│   └── infer.py          # Inference script (loads weights, predicts on new profiles)
├── requirements.txt   # Python dependencies
├── .gitignore         # Files and directories to ignore in Git
└── README.md          # Project overview and instructions

Installation

Clone the repository

git clone https://github.com/<your-username>/sinkhole_cnn.git
cd sinkhole_cnn

Set up a Python virtual environment

python3 -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

Usage

1. Generate Synthetic Data

Creates labeled sinkhole and non-sinkhole elevation profiles:

python src/generate_data.py

The script will output data/synthetic.npz containing two arrays: sinkholes and non_sink.

2. Train the CNN Model

Trains the 1D CNN on the generated dataset and saves the weights to models/sinkhole_cnn.pth:

python src/train.py

3. Run Inference

Loads the trained weights and runs a prediction on example profiles:

python src/infer.py

This will print the sinkhole probability for both a sinkhole-like profile and a non-sink profile.