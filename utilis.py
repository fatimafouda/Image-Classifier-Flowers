import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json

def read_labels(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_data(data_path):
    data_dir = data_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #create transforms
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                             ])
    val_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                                 ])
    test_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                                 ])


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transform)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transform)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(val_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    return trainloader, vloader, testloader, train_data

def create_model(network, nodes, learning_rate, drpout):
    model = getattr(models,network)(pretrained=True)
    model_to_input ={
      "Resnet": 512,
      "Alexnet": 9216,
      "vgg16": 25088,
      "squeezenet1_0": 512,
      "densenet161": 2208,
      "inception_v3": 2048,
      "shufflenet_v2_x1_0": 1024,
      "mobilenet_v2": 1280,
      "resnext50_32x4d": 2048  
    }
    first_layer = model_to_input[network]
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(first_layer, nodes),
                                         nn.ReLU(),
                                         nn.Dropout(drpout),
                                         nn.Linear(nodes, 102),
                                         nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print("Pretrained model type: ",network)
    print("Input layer: ",first_layer)
    print("Nodes in hidden layer: ", nodes)
    print("Output layer: ", 102)
    print("Learning rate: ", learning_rate)
    print("Dropout: ", drpout)
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, epochs, trainloader, vloader, device, train_data):
    model = model
    model.to(device)
    criterion = criterion
    optimizer = optimizer
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 30
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in vloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {val_loss/len(vloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(vloader):.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = train_data.class_to_idx
    accuracy= accuracy/len(vloader)
    print("Finished training")
    print("Accuracy achived: ", round(accuracy,3))
    return model, epoch, accuracy, optimizer

def save_checkpoint(model, model_type, epoch, accuracy, criterion, optimizer, save_path):
    checkpoint = {'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model': model,
            'class_to_idx': model.class_to_idx,
            'criterion': criterion,
            'accuracy': accuracy,
            'model_type': model_type}
    torch.save(checkpoint, save_path)

