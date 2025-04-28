import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from Restocker import Restocker
from val_ai import val
import matplotlib.pyplot as plt

#Data Augmentation and Tranformations to fit training
training_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

#Laoding dataset and applying transformations
train_dataset = datasets.ImageFolder(root="archive (1)/train", transform=training_transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = datasets.ImageFolder(root="archive (1)/validation", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=128)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#Loading Model
model = Restocker()
model.to(device)

#Loss Function
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
lr = 3e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)


train_losses = [] #Used to plot loss graph and visualize loss
val_losses = [] #Used to plot validation loss
acc = [] #Used to plot accuracy

# Training the model and defining train steps
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        curr_loss, current = loss.item(), (batch + 1) * len(X)

        if batch % 10 == 0:
            print(f'loss: {curr_loss:>7f}  [{current:>5d}/{size:>5d}]   LR: {optimizer.param_groups[0]['lr']}')

        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    train_losses.append(avg_loss)


#Training loop
epochs = 50
prev_loss = 0
plateu_counter = 0

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    val_loss, accuracy = val(val_loader, model, loss_fn)

    #Early stopping
    if abs(prev_loss - val_loss) < 0.01:
        plateu_counter += 1
    else:
        plateu_counter = 0
    if plateu_counter == 5:
        print("Plateued so Stopped")
        break
    prev_loss = val_loss

    scheduler.step(val_loss)
    val_losses.append(val_loss)
    acc.append(accuracy)
    if accuracy > 90:
        break

print("Done!")

#Plotting Training Loss
plt.plot(train_losses, label="Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

#plotting Validation Loss
plt.plot(val_losses, label="Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.show()

#Plotting Accuracy
plt.plot(acc, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.show()

torch.save(model.state_dict(), "model.pth")
print("Saved the Model")