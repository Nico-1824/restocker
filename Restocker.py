import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root="archive/train", transform=transform)
val_dataset = datasets.ImageFolder(root="archive/validation", transform=transform)
test_dataset = datasets.ImageFolder(root="archive/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


class Restocker(nn.Module):
    def __init__(self):
        super().__init__()
        #layering goes here for the convolution stack
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), #Looking for features with 16 filters
            nn.ReLU(), #Relu
            nn.MaxPool2d(2), #cuts data in half, 16x64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1), #32x64x64
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
         #finds the size of the flattened output from the convolution step
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self.conv_stack(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.flatten = nn.Flatten() #Flattens the output into a single layer input

        #Linear classification stack
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 36)
        )
    
    #forward propogation
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
########################################################################
# END OF MODEL DEFENITION #
########################################################################
    
model = Restocker()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


train_losses = [] #Used to plot loss graph and visualize loss
# Training the model and defining train steps
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to("cpu"), y.to("cpu")

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    train_losses.append(avg_loss)
        
#Defining the Testing function to see accuracy 
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0 , 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cpu"), y.to("cpu")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

########################################################################
# END OF TRAINING AND TESTING DEFENITION #
########################################################################

#First run of training and testing to see if the model is working 
# properly and if it needs tweaking

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")

import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

