import numpy as np
import time
import sys
import itertools
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Available GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
# Image augmentation
transform_plus = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.RandomAffine(degrees=15, translate=(0.1,0.1))]
                )
transform_norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
transform = transforms.Compose([transforms.ToTensor(),transform_norm])

# Put the downloaded cifar-10 dataset in the data folder under the code directory
trainset0 = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)

testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
# trainset.targets returns the labels of the dataset
# trainset.data returns data shape (50000,32,32,3)
# i.e., 50,000 photos, (32,32,3) represents 32X32 pixels, each pixel has 3 numbers (0-255,0-255,0-255) representing color

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Shuffle dataset
choice=list(range(len(trainset0)))
random.shuffle(choice)

class mydataset(torch.utils.data.Dataset):
    def __init__(self, trainset, choice, num_val, transform=None,transform_norm=None, train=True):
        self.transform = transform
        self.transform_norm = transform_norm
        self.train = train
        self.choice = choice
        self.num_val = num_val
        if self.train:
            self.images = trainset.data[self.choice[self.num_val:]].copy()
            self.labels = [trainset.targets[i] for i in self.choice[self.num_val:]]
        else:
            self.images = trainset.data[self.choice[:self.num_val]].copy()
            self.labels = [trainset.targets[i] for i in self.choice[:self.num_val]]
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        # ToTensor converts array/image to numbers between [0,1]
        image = transforms.ToTensor()(image)
        if self.transform:
            # transforms input is a Tensor
            image = self.transform(image)
        if self.transform_norm:
            image = self.transform_norm(image)
        sample = (image, label)
        return sample

validset = mydataset(trainset0, choice, len(trainset0)//10, None, transform_norm, False)
trainset = mydataset(trainset0, choice, len(trainset0)//10, transform_plus,transform_norm, True)





def test(net, validloader):
    test_loss=0
    test_correct=0
    time=0
    net.eval()
    with torch.no_grad():
        for data in validloader:
            inputs, labels =data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs=net(inputs)
            test_loss += loss_fn(outputs, labels).item()*len(labels)
            test_correct += (torch.max(outputs.data,1)[1] == labels).sum()
            time += 1
    return(test_loss/len(validset), test_correct/len(validset)*100)



class vgg16_conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.4, drop=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3 ,1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop =drop
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.drop:
            x = self.dropout(x)
        return(x)



def vgg16_layer(input_channels, out_channels, num, dropout=[0.4, 0.4]):
    result = []
    result.append(vgg16_conv_block(input_channels, out_channels, dropout[0]))
    for i in range(1, num-1):
        result.append(vgg16_conv_block(out_channels, out_channels, dropout[1]))
    if num>1:
        result.append(vgg16_conv_block(out_channels, out_channels, drop=False))
    result.append(nn.MaxPool2d(2,2))
    return(result)

b1 = nn.Sequential(*vgg16_layer(3,64,2,[0.3,0.4]), *vgg16_layer(64,128,2), *vgg16_layer(128,256,3), 
                   *vgg16_layer(256,512,3),*vgg16_layer(512,512,3))
b2 = nn.Sequential(nn.Dropout(0.5), nn.Flatten(), nn.Linear(512, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(inplace=True), 
                  nn.Linear(512,10, bias=True))
net = nn.Sequential(b1, b2)

# Multi-GPU training setup
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    net = nn.DataParallel(net)
else:
    print("Using single GPU or CPU for training")

net.to(device)
net.train()
epoch_num=200 # Number of epochs
batch_num=128 # Mini-batch size
learning_rate=0.1
train_num=len(trainset)//batch_num
los=[]
cor=[]
train_los=[]
train_cor=[]
net_corr, net_los,net_train_los, net_train_corr, net_lr, net_epoch = 0,0,0, 0,0,0
loss_fn=nn.CrossEntropyLoss()
opt=optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6, nesterov=True)
scheduler3 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=22, T_mult=2)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_num, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_num, shuffle=True)

for epoch in range(epoch_num):
    loss_avg=0 # Average loss
    train_time=0 # Number of mini-batches processed
    correct=0 # Accuracy
    num_img=0
    for data in trainloader:
#     for data in itertools.islice(trainloader, 10):
        inputs, labels = data

        inputs=inputs.to(device)
        labels=labels.to(device)
        
        net.train()
        outputs=net(inputs)
        loss=loss_fn(outputs, labels)
        loss.to(device)
        opt.zero_grad()
        loss.backward()
        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(net.parameters(), 20)
        opt.step()
        train_time+=1
        loss_avg += loss.item()*len(labels)
        # loss.item is the average loss of minibatch
        predict = torch.max(outputs.data,1)[1]
        correct += (predict == labels).sum()
        num_img += len(labels)
        print('\r',end='')
        print('Progress: {} batches'.format(train_time),'',end="")
        sys.stdout.flush()
        
    scheduler3.step() 
    print('\r', end="")
    
    los2,cor2 = test(net, validloader)
    print('Training: {}/{} epochs, Learning rate: {:.10f}, Average Loss: {:.2f}, Accuracy: {:.2f}%, Validation loss: {:.2f}, Validation accuracy: {:.2f}%'
          .format(epoch+1,epoch_num, opt.state_dict()['param_groups'][0]['lr'], loss_avg/num_img, correct/num_img*100,
        los2, cor2.item()))
    
    los.append(los2)
    cor.append(cor2)
    train_cor.append(correct/num_img*100)
    train_los.append(loss_avg/num_img)
    if net_corr < cor2:
        net_corr, net_los,net_train_los, net_train_corr, net_lr, net_epoch = cor2, los2, loss_avg/num_img,correct/num_img,opt.state_dict()['param_groups'][0]['lr'], epoch+1
    torch.save(net, 'net_model.pkl')
    sys.stdout.flush()
print('Model optimal at epoch {}, learning rate: {:.8f}, training loss: {:.4f}, training accuracy: {:.2f}%, validation loss: {:.4f}, validation accuracy: {:.2f}%'.format(net_epoch, net_lr, net_train_los, net_train_corr*100, net_los, net_corr))

try:
    best_net = torch.load('net_model.pkl', map_location=device, weights_only=False)
    print("Successfully loaded complete model")
        
    # If model is wrapped by DataParallel, extract original model
    if isinstance(best_net, nn.DataParallel):
        print("Detected DataParallel model, extracting original model...")
        model = best_net.module
except Exception as e:
    print(f"Failed to load complete model: {e}")

testloader = torch.utils.data.DataLoader(testset, batch_size=100)
test_loss=0
test_correct=0
time=0
best_net.eval()
with torch.no_grad():
    for data in testloader:
        inputs, labels =data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs=best_net(inputs)
        test_loss += loss_fn(outputs, labels).item()*len(labels)
        test_correct += (torch.max(outputs.data,1)[1] == labels).sum()
        time += 1
print('Tested {} images in total, average loss: {:.2f}, accuracy: {:.2f}%'.format(len(testset.data), test_loss/len(testset.data), test_correct/len(testset.data)*100))
x_epoch = [i for i in range(epoch_num)]




# plt.figure()
# plt.plot(x_epoch, train_los,'darkorange')
# plt.plot(x_epoch, los)
# # plt.yscale('log')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train_loss', 'Test_loss'])


# plt.figure()
# plt.plot(x_epoch, torch.tensor(train_cor).cpu())
# plt.plot(x_epoch, torch.tensor(cor).cpu())
# plt.xlabel('Epoch')
# plt.ylabel('Correct')
# plt.legend(['Train_Correct','Test_Correct'])