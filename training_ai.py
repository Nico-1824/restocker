import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from Restocker import Restocker
from val_ai import val
import matplotlib.pyplot as plt

#Z Score Normalization = xi - mean / std ONLY RUN IF NEW DATA ADDED TO DATASET
# temp_transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

# temp_dataset = train_dataset = datasets.ImageFolder(root="archive (1)/train", transform=temp_transform)
# temp_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# mean = 0.
# std = 0.
# num_samples = 0.

# for data, _ in temp_loader:
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     num_samples += batch_samples

# mean /= num_samples
# std /= num_samples

# print(mean, std)

#Data Augmentation and Tranformations to fit training
training_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6137, 0.5516, 0.3981], std=[0.2211, 0.2228, 0.2402])
])



val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6137, 0.5516, 0.3981], std=[0.2211, 0.2228, 0.2402])
])

#Laoding dataset and applying transformations
train_dataset = datasets.ImageFolder(root="archive (1)/train", transform=training_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root="archive (1)/validation", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#Loading Model
model = Restocker()
model.to(device)

#Loss Function
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = StepLR(optimizer, step_size=8, gamma=0.1)


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
            print(f'loss: {curr_loss:>7f}  [{current:>5d}/{size:>5d}]')

        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    train_losses.append(avg_loss)


#Training loop
epochs = 100
prev_loss = 0
plateu_counter = 0

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    with torch.no_grad():
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

    scheduler.step()
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