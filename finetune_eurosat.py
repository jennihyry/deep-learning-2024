"""
Deep learning final project - Fine-tuning with EuroSAT

This is the second part of the project where a pretrained 
ResNet18 model is further trained using the EuroSAT dataset.

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
import random

random.seed(0)

# CHANGE THESE PATHS IF NECESSARY
eurosat_local_path = './EuroSAT/2750'
pretrained_model_path = './state_dict_model.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load EuroSAT data
EuroSAT = torchvision.datasets.ImageFolder(
    root=eurosat_local_path,
)

transform_train = transform=transforms.Compose([
        transforms.Resize((244,244,)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.5, 0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
transform_test = transform=transforms.Compose([
        transforms.Resize((244,244,)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def init_pretrained_model(device, path):
    # Resnet18 model with fc layer changed to match pretrained model
    model = models.resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 64)

    model.to(device)

    # Load model trained on miniImageNet
    model.load_state_dict(torch.load(path, map_location=device))

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the second last layer
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Change the fc layer again to match 10 EuroSAT classes
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    
    return model

def get_sampled_loaders(dataset, batch_size=25):
    # Get unique classes from the dataset
    dataset_classes = dataset.classes

    # Shuffle the classes and select 5 of them
    random.shuffle(dataset_classes)
    selected_classes = dataset_classes[:5]

    # Form test and train sets
    train_indices = []
    test_indices = []

    for class_name in selected_classes: # loops through the randomly selected classes
        class_indices = []
        indices = []
        for idx, label in enumerate(dataset.targets):   # loops through labels in eurosat       
            if dataset.classes[label] == class_name:
                class_indices.append(idx)

        indices = random.sample(class_indices,20)   # contains 20 randomly chosen images from class
        train_indices.extend(indices[:5])           # adds 5 indices
        test_indices.extend(indices[5:])            # adds 15 indices

    # Create train and test sets using the selected indices
    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    
    # Transform
    train_set.dataset.transform = transform_train
    test_set.dataset.transform = transform_test

    # Create a DataLoader for the train and test sets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

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

def train(model, train_loader, valid_loader, lr=0.001, weight_decay=0.01, momentum=0.9, epoches=15):
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

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

    # Train the model
    for epoch in range(epoches):
        model.train()
        correct = 0.0
        num_images = 0.0
        
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
        
        valid_acc_list.append(valid_acc.item())
        train_acc_list.append(train_acc.item())
        losses_list.append(loss.item())
        
        print('epoch: %d, lr: %f, train accuracy: %f, loss: %f, valid accuracy: %f' % (epoch, optimizer.param_groups[0]['lr'], train_acc, loss.item(), valid_acc))
    
    return model, np.mean(train_acc_list)

episodes = 20
test_accuracies = []
train_accuracies = []

for episode in range(episodes):
    
    print(f"\n*** EPISODE {episode+1} ***\n")
    
    # Initialize pretrained model
    model = init_pretrained_model(device, pretrained_model_path)
    
    # Sample data
    train_loader, test_loader = get_sampled_loaders(EuroSAT, batch_size=25)
    
    # Train loop; test_loader is acting as a valid_loader
    model, mean_train_acc = train(model, train_loader, test_loader, epoches=50, lr=0.001, weight_decay=0.1)
    train_accuracies.append(mean_train_acc)
    
    # Test
    test_acc = eval(model, test_loader)
    test_accuracies.append(test_acc.item())
    print(f"Episode {episode+1} test accuracy: {test_acc}")
    
# Average the accuracies
mean_test_acc = np.mean(test_accuracies)
print("\n AVERAGE TEST ACCURACY: ", mean_test_acc)

plt.figure()
plt.title("Test accuracies")
plt.plot(test_accuracies,label="test accuracy")
plt.xlabel("Episodes")
plt.ylabel("Accuracy")
plt.show()