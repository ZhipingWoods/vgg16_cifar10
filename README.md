# VGG16 CIFAR-10 Classifier

A PyTorch implementation of VGG16 deep learning model for CIFAR-10 image classification with comprehensive inference and performance testing capabilities.

## 🚀 Features

- **Modern VGG16 Architecture**: BatchNorm and Dropout layers for improved performance
- **CIFAR-10 Optimized**: Adapted for 32x32 small images  
- **Performance Analysis**: Precise forward/backward propagation timing measurement
- **GPU Accelerated**: Full CUDA support for training and inference

## 📋 Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- CUDA-capable GPU (recommended)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/ZhipingWoods/vgg16_cifar10.git
cd vgg16_cifar10
```

2. **Install dependencies**
```bash
pip install torch torchvision numpy
```

3. **Download pre-trained model**
   - Model weights: [Google Drive Link](https://drive.google.com/file/d/1qS7OCkll7uCzY7u2cak6un4-ExcOAXgC/view?usp=drive_link)
   - Place the `net_model.pkl` file in the project root directory

4. **Prepare CIFAR-10 dataset**
   - Download to `data/cifar-10-batches-py/` directory
   - Or modify the data loading path in the code

## 📁 Project Structure

```
VGG16/
├── README.md                # Project documentation
├── train.py                 # train
├── test.py                  # test
├── test_bp.py               # Main inference bp 
├── net_model.pkl            # Pre-trained model weights
└── data/                    # Data directory
    └── cifar-10-batches-py/ # CIFAR-10 dataset
```

## 🚀 Usage

```bash
python test_bp.py
```

**Configuration**: Modify the paths in `test_bp.py` according to your environment:

```python
cifar10_path = "/your/path/to/cifar-10-batches-py"
model_path = "/your/path/to/net_model.pkl"
```

## 📊 Performance Results

```
Using device: cuda:0
GPU name: NVIDIA GeForce RTX 3090
GPU memory: 23.69 GB

VGG16 CIFAR-10 Model Inference Test
============================================================
Model loading time:          0.164961 seconds
Image loading time:          0.091356 seconds
Forward propagation time:    0.003817 seconds
Backpropagation time (avg):  0.008664 seconds
Pure inference time:         0.003817 seconds
Total time:                  0.260134 seconds

🔥 VGG16 single sample backpropagation time: 0.008664 seconds (Batch Size=1)

Prediction Results:
Predicted class: cat (index: 3)
Confidence: 0.9864 (98.64%)
Prediction correct: Yes

Top-5 Prediction Results:
1.          cat: 0.9864 (98.64%)
2.          dog: 0.0134 (1.34%)
3.         bird: 0.0001 (0.01%)
4.         ship: 0.0000 (0.00%)
5.     airplane: 0.0000 (0.00%)
```

```
Using device: cuda:0
GPU name: NVIDIA A100 80GB PCIe
GPU memory: 79.15 GB

VGG16 CIFAR-10 Model Inference Test
============================================================
Model loading time:          0.129019 seconds
Image loading time:          0.121389 seconds
Forward propagation time:    0.003776 seconds
Backpropagation time (avg):  0.012855 seconds
Pure inference time:         0.003776 seconds
Total time:                  0.254184 seconds

🔥 VGG16 single sample backpropagation time: 0.012855 seconds (Batch Size=1)

Prediction Results:
Predicted class: cat (index: 3)
Confidence: 0.9864 (98.64%)
Prediction correct: Yes

Top-5 Prediction Results:
1.          cat: 0.9864 (98.64%)
2.          dog: 0.0134 (1.34%)
3.         bird: 0.0001 (0.01%)
4.         ship: 0.0000 (0.00%)
5.     airplane: 0.0000 (0.00%)
```



## 🎯 CIFAR-10 Classes

The model recognizes 10 classes:
1. airplane
2. automobile  
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

## 🔧 Model Architecture

- **Feature Extraction**: 5 convolutional blocks with 2-3 conv layers each
- **Classifier**: Fully connected layers with BatchNorm + ReLU + Dropout
- **Output**: 10-class softmax classifier optimized for CIFAR-10 32x32 images


---

**Note**: Download the pre-trained model weights before first run.