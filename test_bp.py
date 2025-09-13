import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import pickle
import os


class vgg16_conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.4, drop=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop = drop
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.drop:
            x = self.dropout(x)
        return x

def vgg16_layer(input_channels, out_channels, num, dropout=[0.4, 0.4]):
    result = []
    result.append(vgg16_conv_block(input_channels, out_channels, dropout[0]))
    for i in range(1, num-1):
        result.append(vgg16_conv_block(out_channels, out_channels, dropout[1]))
    if num > 1:
        result.append(vgg16_conv_block(out_channels, out_channels, drop=False))
    result.append(nn.MaxPool2d(2, 2))
    return result

def create_infer_model():
    # Feature extraction part
    b1 = nn.Sequential(*vgg16_layer(3,64,2,[0.3,0.4]), *vgg16_layer(64,128,2), *vgg16_layer(128,256,3), 
                       *vgg16_layer(256,512,3), *vgg16_layer(512,512,3))
    # Classifier part
    b2 = nn.Sequential(nn.Dropout(0.5), nn.Flatten(), nn.Linear(512, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(inplace=True), 
                      nn.Linear(512, 10, bias=True))
    # Complete network
    net = nn.Sequential(b1, b2)
    return net




# Device configuration - Force using single GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Specify using first GPU
    torch.cuda.set_device(0)  # Set current GPU
    print(f"Using device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")
else:
    raise RuntimeError("No available GPU detected, but the program requires GPU for inference")

# Path configuration
# ======================
# you need change these paths according to your environment
cifar10_path = "/home/wuzhiping/code/VGG16/data/cifar-10-batches-py"
model_path = "/home/wuzhiping/code/VGG16/net_model.pkl"
# ======================

# CIFAR-10 class names
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_cifar10_batch(file_path):
    """
    Load CIFAR-10 data batch file
    """
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    
    # Convert data format
    data = batch[b'data']
    labels = batch[b'labels']
    
    # Reshape data to (batch_size, 3, 32, 32) format
    data = data.reshape(len(data), 3, 32, 32)
    data = data.astype(np.float32) / 255.0  # Normalize to [0,1]
    
    return data, labels

def load_model(model_path, device):
    """
    Load trained model
    """
    print(f"Loading model from {model_path}...")
    start_time = time.time()
    
    try:
        # Try to load the complete model directly, set weights_only=False
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("Successfully loaded complete model")
        
        # If model is wrapped by DataParallel, extract original model
        if isinstance(model, nn.DataParallel):
            print("Detected DataParallel model, extracting original model...")
            model = model.module
            
    except Exception as e:
        print(f"Direct loading failed: {e}")
        print("Trying to create new model and load weights...")
        model = create_infer_model()
        
        try:
            # Try to load weights
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, nn.DataParallel):
                # If checkpoint is DataParallel wrapped model
                state_dict = checkpoint.module.state_dict()
            elif hasattr(checkpoint, 'state_dict'):
                # If checkpoint has state_dict method
                state_dict = checkpoint.state_dict()
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # If checkpoint is dict and contains state_dict
                state_dict = checkpoint['state_dict']
            else:
                # Use checkpoint directly as state_dict
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            print("Successfully loaded weights to newly created model")
            
        except Exception as load_error:
            print(f"Failed to load weights: {load_error}")
            raise load_error
    
    # Ensure model is on correct device
    model = model.to(device)
    model.eval()
    
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.4f} seconds")
    
    return model, load_time

def preprocess_image(image_data):
    """
    Preprocess image data
    """
    # Apply same normalization as during training
    transform_norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                        std=(0.2023, 0.1994, 0.2010))
    
    # Convert to Tensor and normalize
    image_tensor = torch.from_numpy(image_data).float()
    image_tensor = transform_norm(image_tensor)
    
    return image_tensor

def inference_with_timing(model, image_tensor, device):
    """
    Execute inference and measure timing - Focus on backpropagation timing measurement
    """
    # Ensure data is on GPU
    image_tensor = image_tensor.unsqueeze(0).to(device)
    print(f"Data loaded to device: {image_tensor.device}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Warm up GPU (more thorough warming)
    print("Warming up GPU...")
    model.eval()  # Use eval mode to avoid BatchNorm issues
    for _ in range(10):
        temp_input = image_tensor.clone().detach().requires_grad_(True)
        temp_output = model(temp_input)
        temp_loss = temp_output.sum()
        temp_loss.backward()
        model.zero_grad()
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # 1. Measure pure forward propagation time
    model.eval()
    torch.cuda.synchronize()
    forward_start = time.time()
    
    with torch.no_grad():
        outputs_forward = model(image_tensor)
    
    torch.cuda.synchronize()
    forward_time = time.time() - forward_start
    
    # 2. Measure backpropagation time (batch_size=1)
    # To solve BatchNorm issues with batch_size=1, we need special handling
    model.eval()  # Set to eval mode first
    # But need to enable gradient computation for backpropagation
    
    # Multiple measurements of backpropagation time for averaging
    backward_times = []
    num_backward_runs = 10
    
    for i in range(num_backward_runs):
        # Create new input copy, ensure gradient computation
        input_with_grad = image_tensor.clone().detach().requires_grad_(True)
        
        # Zero gradients
        model.zero_grad()
        if input_with_grad.grad is not None:
            input_with_grad.grad.zero_()
        
        torch.cuda.synchronize()
        backward_start = time.time()
        
        # Forward propagation (eval mode but with gradient enabled)
        outputs = model(input_with_grad)
        # Use cross-entropy loss function (more realistic backpropagation scenario)
        loss = nn.CrossEntropyLoss()(outputs, torch.tensor([0], device=device))  # Assume label is 0
        
        # Backpropagation
        loss.backward()
        
        torch.cuda.synchronize()
        backward_time_single = time.time() - backward_start
        backward_times.append(backward_time_single)
    
    # Calculate backpropagation statistics
    backward_times = np.array(backward_times)
    avg_backward_time = backward_times.mean()
    min_backward_time = backward_times.min()
    max_backward_time = backward_times.max()
    std_backward_time = backward_times.std()
    
    # Get prediction results
    model.eval()
    with torch.no_grad():
        final_outputs = model(image_tensor)
        _, predicted = torch.max(final_outputs.data, 1)
        probabilities = torch.softmax(final_outputs, dim=1)
    
    return (final_outputs, predicted, probabilities, forward_time, 
            avg_backward_time, min_backward_time, max_backward_time, std_backward_time)

def main():
    """
    Main function
    """
    print("="*60)
    print("VGG16 CIFAR-10 Model Inference Test")
    print("="*60)
    
    model_path = "/home/wuzhiping/code/VGG16/net_model.pkl"
    
    # 1. Load model
    model, model_load_time = load_model(model_path, device)
    
    # 2. Load test image
    print("\nLoading CIFAR-10 test data...")
    image_load_start = time.time()
    
    # Load test batch
    test_batch_path = os.path.join(cifar10_path, "test_batch")
    if not os.path.exists(test_batch_path):
        print(f"Error: Cannot find test data file {test_batch_path}")
        return
    
    data, labels = load_cifar10_batch(test_batch_path)
    
    # Select first image for testing
    test_image = data[0]  # Shape: (3, 32, 32)
    true_label = labels[0]
    
    # Preprocess image
    processed_image = preprocess_image(test_image)
    image_load_time = time.time() - image_load_start
    
    print(f"Image loading and preprocessing time: {image_load_time:.4f} seconds")
    print(f"Test image shape: {test_image.shape}")
    print(f"True label: {classes[true_label]} (index: {true_label})")
    
    # 3. Execute inference
    print("\nStarting inference...")
    (outputs, predicted, probabilities, forward_time, 
     avg_backward_time, min_backward_time, max_backward_time, std_backward_time) = inference_with_timing(
        model, processed_image, device
    )
    
    # 4. Display results
    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item()
    
    print("\n" + "="*40)
    print("Inference Results:")
    print("="*40)
    print(f"Predicted class: {classes[predicted_class]} (index: {predicted_class})")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"Prediction correct: {'Yes' if predicted_class == true_label else 'No'}")
    
    # 5. Time statistics - Focus on backpropagation time
    total_inference_time = forward_time
    total_time = model_load_time + image_load_time + total_inference_time
    
    print("\n" + "="*40)
    print("Time Statistics:")
    print("="*40)
    print(f"Model loading time:          {model_load_time:.6f} seconds")
    print(f"Image loading time:          {image_load_time:.6f} seconds")
    print(f"Forward propagation time:    {forward_time:.6f} seconds")
    print(f"Backpropagation time (avg):  {avg_backward_time:.6f} seconds")
    print(f"Pure inference time:         {total_inference_time:.6f} seconds")
    print(f"Total time:                  {total_time:.6f} seconds")
    print(f"\n🔥 VGG16 single sample backpropagation time: {avg_backward_time:.6f} seconds (Batch Size=1)")
    
    # 6. Display top-5 predictions
    print("\n" + "="*40)
    print("Top-5 Prediction Results:")
    print("="*40)
    top5_prob, top5_indices = torch.topk(probabilities[0], 5)
    for i in range(5):
        idx = top5_indices[i].item()
        prob = top5_prob[i].item()
        print(f"{i+1}. {classes[idx]:>12}: {prob:.4f} ({prob*100:.2f}%)")
    
    # # 7. Multiple inference performance test
    # print("\n" + "="*40)
    # print("Performance Benchmark Test (100 inferences):")
    # print("="*40)
    
    # num_runs = 100
    # times = []
    
    # with torch.no_grad():
    #     # Warm up
    #     for _ in range(10):
    #         _ = model(processed_image.unsqueeze(0).to(device))
        
    #     # Actual test
    #     for i in range(num_runs):
    #         torch.cuda.synchronize() if torch.cuda.is_available() else None
    #         start = time.time()
    #         _ = model(processed_image.unsqueeze(0).to(device))
    #         torch.cuda.synchronize() if torch.cuda.is_available() else None
    #         times.append(time.time() - start)
    
    # times = np.array(times)
    # print(f"Average inference time:      {times.mean():.6f} seconds")

if __name__ == "__main__":
    main()