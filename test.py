import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Available GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Data preprocessing - same as training
transform_norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
transform = transforms.Compose([transforms.ToTensor(), transform_norm])

# Load CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# CIFAR-10 class names
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# VGG16 model architecture - same as training
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

def create_vgg16_model():
    """Create VGG16 model with same architecture as training"""
    b1 = nn.Sequential(*vgg16_layer(3,64,2,[0.3,0.4]), *vgg16_layer(64,128,2), *vgg16_layer(128,256,3), 
                       *vgg16_layer(256,512,3), *vgg16_layer(512,512,3))
    b2 = nn.Sequential(nn.Dropout(0.5), nn.Flatten(), nn.Linear(512, 512, bias=True), 
                      nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, 10, bias=True))
    net = nn.Sequential(b1, b2)
    return net

def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    try:
        # Try to load complete model
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("Successfully loaded complete model")
        
        # If model is wrapped by DataParallel, extract original model
        if isinstance(model, nn.DataParallel):
            print("Detected DataParallel model, extracting original model...")
            model = model.module
            
    except Exception as e:
        print(f"Failed to load complete model: {e}")
        print("Trying to create new model and load weights...")
        
        # Create model and try to load weights
        model = create_vgg16_model()
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, nn.DataParallel):
                state_dict = checkpoint.module.state_dict()
            elif hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            print("Successfully loaded weights to newly created model")
        except Exception as load_error:
            print(f"Failed to load weights: {load_error}")
            return None
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, testloader):
    """Evaluate model on test dataset"""
    model.eval()
    
    test_loss = 0
    test_correct = 0
    total_samples = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    loss_fn = nn.CrossEntropyLoss()
    
    print("Starting evaluation...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            test_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            total_samples += len(labels)
            
            # Calculate per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f'Progress: {batch_idx}/{len(testloader)} batches processed')
    
    end_time = time.time()
    
    # Calculate overall metrics
    avg_loss = test_loss / total_samples
    accuracy = 100. * test_correct / total_samples
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total test samples: {total_samples}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Evaluation time: {end_time - start_time:.2f} seconds")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    print("-" * 30)
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{classes[i]:>12}: {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            print(f'{classes[i]:>12}: N/A (no samples)')
    
    return avg_loss, accuracy, class_correct, class_total

def inference_speed_test(model, testloader, num_batches=50):
    """Test inference speed"""
    print(f"\nRunning inference speed test with {num_batches} batches...")
    
    model.eval()
    times = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            if batch_idx >= num_batches:
                break
                
            inputs = inputs.to(device)
            
            # Warm up
            if batch_idx < 5:
                _ = model(inputs)
                continue
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            outputs = model(inputs)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
    
    times = np.array(times)
    avg_time = times.mean()
    std_time = times.std()
    
    print(f"Average inference time per batch: {avg_time:.6f} ± {std_time:.6f} seconds")
    print(f"Average inference time per sample: {avg_time/100:.6f} seconds")
    print(f"Inference throughput: {100/avg_time:.2f} samples/second")

def visualize_predictions(model, testloader, num_samples=8):
    """Visualize some predictions"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(testloader)
    images, labels = next(data_iter)
    
    # Select random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    sample_images = images[indices]
    sample_labels = labels[indices]
    
    # Make predictions
    with torch.no_grad():
        sample_images = sample_images.to(device)
        outputs = model(sample_images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Unnormalize image for display
        img = sample_images[i].cpu()
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = classes[sample_labels[i]]
        pred_label = classes[predicted[i]]
        confidence = probabilities[i][predicted[i]].item()
        
        color = 'green' if predicted[i] == sample_labels[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                         color=color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=150, bbox_inches='tight')
    print(f"\nPrediction visualization saved as 'prediction_samples.png'")
    plt.show()

def main():
    """Main testing function"""
    print("="*60)
    print("VGG16 CIFAR-10 Model Testing")
    print("="*60)
    
    # Model path
    model_path = 'net_model.pkl'
    
    # Load model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    # Evaluate model
    avg_loss, accuracy, class_correct, class_total = evaluate_model(model, testloader)
    
    # Speed test
    inference_speed_test(model, testloader)
    
    # Visualize some predictions
    try:
        visualize_predictions(model, testloader)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()