# Waste Classification System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## TL;DR

End-to-end computer vision pipeline for waste classification using a custom dataset, EfficientNet-based feature extraction, and ONNX runtime for optimized edge inference. Designed for real-world deployment with hardware integration and scalable system architecture.

---

## 🚀 Demo

Live working demo:

👉 https://drive.google.com/file/d/1cU7xLgWyejoR9gFWWKvO4Yxo-BgdY4as/view?usp=sharing

---

## 📌 Overview

This project is a complete **waste classification system** built using deep learning and designed for **real-world deployment**.

It combines:
- Custom dataset creation  
- Model training and optimization  
- ONNX-based inference pipeline  
- Edge deployment compatibility  
- Hardware integration (Raspberry Pi + sensors + servo system)  
- User and municipal-level application support  

---

## 🧠 Features

- Custom dataset built for real-world waste scenarios  
- Multi-class classification (plastic, paper, glass, metal, organic, e-waste)  
- EfficientNet-based model  
- ONNX optimized inference  
- Runs on CPU (edge-friendly)  
- Sensor-triggered automation pipeline  
- Configurable bin structure (hospitals, offices, schools)  
- Geo-location tracking via user + municipal apps  
- End-to-end system design (ML + hardware + application)  

---

## 📂 Project Structure
waste-classification-system/

│
├── src/

│ ├── inference.py

│ ├── test.jpg
│

├── weights/

│ ├── visual_model.onnx

│ ├── text_features.npy

│
├── demo/


│ ├── demo_link.txt

│

├── smart_bin_presentation.pptx

├── README.md

├── requirements.txt



---

## ⚙️ Installation & Setup

### 1. Clone Repository

git clone https://github.com/Pavitraraman/waste-classification-system.git
cd waste-classification-system

2. Create Virtual Environment

python -m venv venv

Activate

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate


4. Install Dependencies

pip install torch torchvision timm numpy matplotlib seaborn scikit-learn tqdm opencv-python onnxruntime

▶️ Running the Model

Option 1 (from project root)

python src/inference.py --image src/test.jpg

Option 2 (from inside src folder)

python inference.py --image test.jpg

📊 Output Example

Prediction: plastic (0.87)

🧪 Model Details

Architecture: EfficientNet-based CNN

Format: ONNX (optimized inference)

Training Accuracy: ~96%

Confidence Range: ~80%

Evaluation: Confusion matrix

---
🔮 Future Work

YOLO-based object detection

Real-time video classification

Edge optimization (quantization / TensorRT)

Full-stack app integration

Smart routing for waste collection
