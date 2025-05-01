import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR
from Restocker import Restocker
from testing_ai import test
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(0.1),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root="archive/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.ImageFolder(root="archive/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64)

model = Restocker()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


train_losses = [] #Used to plot loss graph and visualize loss
val_losses = [] #Used to plot validation loss

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

#First run of training and testing to see if the model is working 
# properly and if it needs tweaking

epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    val_loss, accuracy = test(test_loader, model, loss_fn)
    val_losses.append(val_loss)
    scheduler.step()
    if accuracy > 95:
        break

print("Done!")

plt.plot(train_losses, label="Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(val_losses, label="Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

torch.save(model.state_dict(), "model.pth")
print("Saved the Model")