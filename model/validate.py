import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Restocker import Restocker
import random

model = Restocker().to('cpu')
model.load_state_dict(torch.load("model/model.pth", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

val_dataset = datasets.ImageFolder(root="archive/validation", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32)

for i in range(10):
    image, label = val_dataset[random.randint(0, len(val_dataset) - 1)]
    # Convert tensor to numpy array and adjust dimensions for matplotlib
    image_print = image.permute(1, 2, 0).numpy()
    plt.imshow(image_print)
    plt.show()

    # Add batch dimension and get prediction
    image = image.unsqueeze(0)  # Add batch dimension
    pred = model(image)
    pred_class = torch.argmax(pred, dim=1).item()
    class_name = val_dataset.classes[pred_class]

    print(f"This is a {class_name}")
    print(f"True label: {val_dataset.classes[label]}")