import os
import torch
import torch.nn as nn
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import urllib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

DATADIR = "./dataset"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

# mini-batch size for the training
batch_size = 10

# Create the training and validation data sets
# and the corresponding data loaders.
train_dir = os.path.join(DATADIR, "train")

# Differents cats we want to identify. Each cat has its folder in the training set (labelization)
classes = ("Billy", "Maya", "Nara", "Neko")

# load the whole dataset
dataset = datasets.ImageFolder(train_dir, transform=transform)

# split the dataset with 80% for training and 20% for evaluation
train_number = int(np.round(0.8 * len(dataset)))
valid_number =  len(dataset) - train_number

train_set, valid_set = torch.utils.data.random_split(dataset, [train_number, valid_number])

# create the loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=6)

# load preexisting model mobilenet v2
model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)

# freeze feature layers as their weight might be good enough
for feature in model.features:
    feature.requires_grad = False

# create the new classifier layer we want to apply to the model
classifier = nn.Sequential(
    #nn.Dropout(p=0.1, inplace=False),
    nn.Linear(in_features=1280, out_features=4, bias=True)
    #nn.ReLU(inplace=True),
    #nn.Dropout(p=0.1, inplace=False),
    #nn.Linear(in_features=100, out_features=4, bias=True)
)
model.classifier = classifier

# let's have a look at the model
print(model)

# transfer the model to the GPU if available (hope it is for performance reasons !)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("retained device is ", device)
model = model.to(device)

# fix the learning rate and the momentum of the optimizer
learning_rate = 0.001
momentum = 0.9

# the optimizer is on the device where the model has been put on
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# create the optimizer (loss function) and put on the computing device
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)


def train(net, loader, criterion, optimizer, device):
    """Perform one epoch over the input set.

    Arguments:
      net : Input neural network
      loader : Data loader
      criterion : function to compute the loss
      optimizer :
      device : The device on which to run the computation

    """
    # Set the network to "training" mode
    net.train()

    total_loss = 0
    n_correct = 0

    for i, (batch, target) in enumerate(loader):
        # Move both batch and target to the GPU
        # on the same device as the model and the
        # criterion.
        #
        # Don't forget that tensor needs assignment
        # (i.e. you need to do x = x.to(device), not
        # just x.to(device) like with a model)
        batch = batch.to(device)
        target = target.to(device)

        # 1. Perform the forward pass on the mini-batch
        output = net.forward(batch)

        n_correct += count_nb_correct(output, target)

        # 2. Compute the loss (variable 'loss')
        loss = criterion(output, target)

        # 3. Run the backward pass :
        # a. Zero the values of the optimizer
        optimizer.zero_grad()
        # b. Compute the loss backward
        loss.backward()
        # c. Perform one step of optimizer
        optimizer.step()

        total_loss += loss.item()

    return total_loss, float(n_correct) / float(len(loader.dataset))


def count_nb_correct(output, target):
    """Count the number of correct labels compared to
    the ground truth

    Args:
        output: The output of a network, a tensor of size (n_samples, n_classes)
        target: The ground truth, returned by the DataLoader, a tensor of size (n_samples, )

    Returns:
        The number of correct labels

    """
    return torch.eq(torch.argmax(output, axis=1), target).sum().item()


def validate(net, loader, device):
    # Set the network to evalution mode
    net.eval()

    # Total number of correctly classified samples
    n_correct = 0

    with torch.no_grad():
        for i, (batch, target) in enumerate(loader):
            # Move both batch and target to the GPU
            # on the same device as the model and the
            # criterion.
            batch = batch.to(device)
            target = target.to(device)

            # 1. Perform the forward pass
            output = net.forward(batch)

            # 2. Update 'n_correct'
            n_correct += count_nb_correct(output, target)

    return float(n_correct) / float(len(loader.dataset))


def plot_results(epochs, train_losses, valid_accuracies, figsize=(16,5)):
    """
    """
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "r", label="Train loss")
    plt.title("Train loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_accuracies, "r", label="Validation accuracy")
    plt.title("Validation accuracy")
    plt.legend()
    plt.show()


n_epochs = 10
epochs = []
train_losses = []
train_accuracies = []
valid_accuracies = []

print(f"The training is done on the {'GPU' if next(model.parameters()).is_cuda else 'CPU'}")
beg = time.perf_counter()
for epoch in range(n_epochs):
    # Perform training
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    epochs.append(epoch)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)
    # Test on validation set
    valid_accuracy = validate(model, valid_loader, device)
    valid_accuracies.append(valid_accuracy)
    print(f"epoch {epoch} loss {train_loss:.3f} train accuracy {100*train_accuracy:.2f}% validation accuracy {100*valid_accuracy:.2f}%")
print(f"Training took {time.perf_counter()-beg:.2f} seconds")
plot_results(epochs, train_losses, valid_accuracies)


def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


model_save('torch_cat.model')