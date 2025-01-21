# trainplotkit
Create live subplots in your notebook that update while training a PyTorch model

# Features
* Extensible framework for adding subplots updated in real time in your notebook during training
* Interaction between subplots after training has completed
  * Click on one subplot to select an epoch / sample and update other subplots dynamically
* Supports custom training loops and high-level training libraries like [pytorch_lightning](https://github.com/Lightning-AI/pytorch-lightning) and [fastai](https://github.com/fastai/fastai)
  * Coming soon: adapters for even more seamless integration with high-level training libraries
* All graph interactions provided by [plotly](https://plotly.com/python/)
* Built-in subplots:
  * Training curves
  * Custom metric vs epoch
  * Validation loss for individual samples (scatter plot)
  * Input image corresponding to selected sample
  * Class probililities corresponding to selected sample
  * Coming soon: colourful dimension plot from [fastai course Lesson 16](https://course.fast.ai/Lessons/lesson16.html) 1:14:30 for visualizing activation stats

# Use cases
* Quickly identifying and explaining outlier samples in a dataset
* Quickly developing visualizations to improve your understanding of a model and/or training process

# Installation
```
pip install git+https://github.com/d112358/trainplotkit.git
```

# Usage example
```python
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets, models
from torcheval.metrics import MulticlassAccuracy
from trainplotkit.plotgrid import PlotGrid
from trainplotkit.subplots.basic import TrainingCurveSP, MetricSP, ValidLossSP, ImageSP, ClassProbsSP

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
transform    = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
train_data   = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
valid_data   = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, num_workers=15, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=64, num_workers=15, shuffle=False)
num_classes  = len(valid_data.classes)

# Model setup
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Plots
batch_loss_fn = nn.CrossEntropyLoss(reduction='none')
probs_fn = lambda preds: torch.softmax(preds, dim=1)
sps = [
    TrainingCurveSP(colspan=2), 
    ValidLossSP(batch_loss_fn, remember_past_epochs=True, colspan=2), 
    ImageSP(valid_data, class_names=valid_data.classes, rowspan=2),
    MetricSP("Accuracy", MulticlassAccuracy(), colspan=2), 
    ClassProbsSP(probs_fn, remember_past_epochs=True, class_names=valid_data.classes, colspan=2),
]
pg = PlotGrid(num_grid_cols=5, subplots=sps)
pg.show()

# Training and validation loop
for epoch in range(4):
    # Training
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pg.after_batch(training=True, inputs=images, targets=labels, predictions=outputs, loss=loss)
    pg.after_epoch(training=True)

    # Validation
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            pg.after_batch(training=False, inputs=images, targets=labels, predictions=outputs, loss=loss)
    pg.after_epoch(training=False)
pg.after_fit()
```
![Usage example](https://github.com/d112358/trainplotkit/raw/main/images/usage_example.png)

# License
This repository is released under the MIT license. See [LICENSE](https://github.com/d112358/trainplotkit/blob/main/LICENSE) for additional details.
