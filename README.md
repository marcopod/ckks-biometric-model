# CKKS Biometric Model

**Privacy-Preserving Fingerprint Recognition Using Homomorphic Encryption**

This repository contains the core implementation of a secure fingerprint recognition pipeline that leverages **CKKS homomorphic encryption** and **neural networks** to perform inference on encrypted fingerprint data. The project is designed to integrate with high-resolution tactile inputs from Meta’s DIGIT sensor.

---

## Overview

This project explores the use of encrypted neural network inference for biometric authentication, ensuring data privacy during the entire inference process.

Key components:
- **CKKS-based encryption** using [TenSEAL](https://github.com/OpenMined/TenSEAL)
- **CNN model** trained on plaintext, tested on encrypted data
- **Integration-ready** with external tactile sensing libraries such as [`digit-sensor-lib`](https://github.com/your-org/digit-sensor-lib)
- Performance metrics and reproducible experiments

---

## Repository Structure

\```
ckks-biometric-model/
├── biometric_model/           # Core model, encryption utilities, inference logic
│   ├── model.py
│   ├── ckks_context.py
│   ├── encrypt_utils.py
│   └── inference.py
├── experiments/               # Scripts for testing with MNIST and DIGIT data
├── data/                      # Storage for encrypted and processed data
├── notebooks/                 # Experimentation and visualization
├── configs/                   # Configuration files for models and encryption
├── tests/                     # Unit and integration tests
├── requirements.txt
└── README.md
\```

---

## Requirements

- Python 3.8+
- TenSEAL
- PyTorch
- NumPy
- (Optional) digit-sensor-lib

Install dependencies:

\```bash
pip install -r requirements.txt
\```

---

## Model Pipeline

1. **Plaintext training**: Train a CNN on fingerprint data (e.g., MNIST or DIGIT preprocessed).
2. **Encryption**: Encrypt inputs and model weights using CKKS.
3. **Inference**: Run homomorphic inference on encrypted data.
4. **Decryption**: Output decrypted predictions.

---

## Quick Start

\```python
from biometric_model import ckks_context, encrypt_utils, inference
from digit_sensor import DigitSensor

# Initialize CKKS context
context = ckks_context.create_context()

# Load and vectorize fingerprint
sensor = DigitSensor()
image = sensor.capture()
vector = sensor.vectorize(image)

# Encrypt and infer
enc_input = encrypt_utils.encrypt_input(vector, context)
prediction = inference.run_encrypted_inference(enc_input, context)
\```

---

## Experiments

Run MNIST encrypted inference:
\```bash
python experiments/test_mnist_ckks.py
\```

Run DIGIT-based fingerprint test:
\```bash
python experiments/test_digit_ckks.py
\```

---

## Acknowledgments

- Meta AI - DIGIT tactile sensor