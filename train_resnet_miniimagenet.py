"""
Deep Learning - Final project.

This is the first part where a ResNet18 model is trained using miniImageNet dataset. 
The initial training data is split further to train, validation and test sets.

"""

# import libraries
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("Load data")

data_path = './train'

miniimagenet = torchvision.datasets.ImageFolder(root=data_path)

transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# Split data into train, val and test sets
train_size = int(0.7 * len(miniimagenet))  # 70% for training
val_size = int(0.15 * len(miniimagenet))   # 15% for validation
test_size = int(0.15 * len(miniimagenet))  # 15% for test

train_set, val_set, test_set = torch.utils.data.random_split(miniimagenet, [train_size, val_size, test_size])

# Transforms
train_set.dataset.transform = transform_train
val_set.dataset.transform = transform
test_set.dataset.transform = transform

# Dataloaders
batch_size = 64

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

print("Data split done")

epoches = 15
lr = 0.001
num_classes = 64

# Model: resnet18
model = models.resnet18(weights='DEFAULT')

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Change fc layer and output layer sizes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)

# Loss function
loss_function = torch.nn.CrossEntropyLoss()

# Check for Cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()

# Lists for accuracies and losses
train_acc_list = []
valid_acc_list = []
losses_list = []

def eval(net, data_loader):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    net.eval()
    correct = 0.0
    num_images = 0.0
    for i_batch, (images, labels) in enumerate(data_loader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        outs = net(images) 
        preds = outs.argmax(dim=1)
        correct += preds.eq(labels).sum()
        num_images += len(labels)

    acc = correct / num_images
    return acc

# Train the model
print("Start training")

for epoch in range(epoches):
    model.train()
    correct = 0.0
    num_images = 0.0
    
    # Loop through the train_loader to gather few-shot samples
    for i_batch, (images, labels) in enumerate(train_loader):
        if use_cuda: 
            images = images.cuda()
            labels = labels.cuda()
        # zero grad
        optimizer.zero_grad()
        
        # forward pass
        predictions = model(images)
        
        # loss 
        loss = loss_function(predictions, labels)
        
        # backward pass
        loss.backward()
        
        # gradient descent
        optimizer.step()
        
        # update variables correct and num_images
        prob = F.softmax(predictions, dim=1)
        f = prob.data.argmax(dim=1)
        correct += f.eq(labels.data).sum()
        num_images += len(labels)
        
    # calculate accuracy
    train_acc = correct / num_images
    
    valid_acc = eval(model, valid_loader)

    train_acc_list.append(train_acc.item())
    valid_acc_list.append(valid_acc.item())
    losses_list.append(loss.item())
    
    print('epoch: %d, lr: %f, train accuracy: %f, loss: %f, valid accuracy: %f' % (epoch, optimizer.param_groups[0]['lr'], train_acc, loss.item(), valid_acc))

# Save the trained model

print("Training done!")

test_acc = eval(model, test_loader)

print(f"\nTESTING ACCURACY: {test_acc}\n")

print(f"\nTRAINING ACCURACY LIST: {train_acc_list}")
print(f"\nVALIDATION ACCURACY LIST: {valid_acc_list}")
print(f"\nLOSSES LIST: {losses_list}")

plt.figure()
plt.title("Training and validation accuracies")
plt.plot(train_acc_list,label="train accuracy")
plt.plot(valid_acc_list,label="validation accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.title("Training losses")
plt.plot(losses_list)
plt.xlabel("Iterations")
plt.ylabel("Losses")
plt.show()

path = "./state_dict_model.pt"

torch.save(model.state_dict(), path)

print(f"Model saved to {path}")