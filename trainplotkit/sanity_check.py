import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
import torchvision.transforms as T
from torchvision import datasets, models
import numpy as np
from torcheval.metrics import MulticlassAccuracy
from .plotgrid import PlotGrid
from .subplots.basic import TrainingCurveSP, MetricSP, ValidLossSP, ImageSP, ClassProbsSP

def setup_sanity_check():
    """
    Execute most features from this toolbox to test whether an installation 
    was successful and/or whether any core features were broken during
    an update.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    np.random.seed(42)
    sample_fraction = 0.2  # CIFAR-10 has 6000 images per class for each of 10 classes
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    full_train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    full_valid_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_indices   = np.random.choice(len(full_train_data), size=int(sample_fraction * len(full_train_data)), replace=False)
    valid_indices   = np.random.choice(len(full_valid_data), size=int(sample_fraction * len(full_valid_data)), replace=False)
    train_data      = Subset(full_train_data, train_indices)
    valid_data      = Subset(full_valid_data, valid_indices)
    train_loader    = DataLoader(train_data, batch_size=64, num_workers=15, shuffle=True)
    valid_loader    = DataLoader(valid_data, batch_size=64, num_workers=15, shuffle=False)
    num_classes     = len(full_valid_data.classes)

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
        ImageSP(valid_data, class_names=full_valid_data.classes, rowspan=2),
        MetricSP("Accuracy", MulticlassAccuracy(), colspan=2), 
        ClassProbsSP(probs_fn, remember_past_epochs=True, class_names=full_valid_data.classes, colspan=2),
    ]
    pg = PlotGrid(num_grid_cols=5, subplots=sps)
    return train_loader, valid_loader, model, optimizer, criterion, device, pg

def run_sanity_check(train_loader, valid_loader, model, optimizer, criterion, device, pg):
    # Training and validation loop
    pg.before_fit()
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

        # Validation step
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
    return pg