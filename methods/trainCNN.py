import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import datetime
import time
from PIL import Image
import traceback
import sys


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_model(model_name, n_classes, pretrained=True):
    if model_name == 'InceptionV3':
        model = models.inception_v3(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name == 'VGG16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name == 'Xception':
        model = models.xception(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError("Model name not recognized.")
    return model


def preprocess_input(image, model_name):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return preprocess(image)


def get_transform(model_name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_model(model, output_dir, model_name):
    model_path = os.path.join(output_dir, f'{model_name}_model.pth')
    torch.save(model.state_dict(), model_path)


def load_data(x_path, y_path):
    if x_path.endswith('.npy'):
        x_data = np.load(x_path)
    else:
        raise ValueError("Only .npy format is supported for x data")

    if y_path.endswith('.npy'):
        y_data = np.load(y_path)
    elif y_path.endswith('.csv'):
        y_data = pd.read_csv(y_path).values
    else:
        raise ValueError("Only .npy and .csv formats are supported for y data")

    return x_data, y_data


def main_driver(x_train_path, y_train_path, x_val_path, y_val_path, x_test_path, output_path, model_name, pretrained, aug, d, note):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    x_train, y_train = load_data(x_train_path, y_train_path)
    x_val, y_val = load_data(x_val_path, y_val_path) if x_val_path and y_val_path else (None, None)
    x_test = np.load(x_test_path) if x_test_path else None

    # Transform data
    transform = get_transform(model_name)
    train_dataset = CustomDataset(x_train, y_train, transform=transform)
    val_dataset = CustomDataset(x_val, y_val, transform=transform) if x_val is not None else None

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4) if val_dataset else None

    dataloaders = {'train': train_loader, 'val': val_loader}

    # Model, criterion, optimizer, scheduler
    model = get_model(model_name, n_classes=4, pretrained=pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device=device)

    # Save model
    save_model(model, output_path, model_name)


def get_parser():
    """Defines the parser for this script."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-xtrn", dest="x_train_path", type=str, help="x train path")
    parser.add_argument("-xval", dest="x_val_path", type=str, default='', help="x val path")
    parser.add_argument("-ytrn", dest="y_train_path", type=str, help="y train path ")
    parser.add_argument("-yval", dest="y_val_path", type=str, default='', help="y val path")
    parser.add_argument("-xtest", dest="x_test_path", type=str, default=None, help='x test path')
    parser.add_argument("-o", dest="output_path", type=str, help='base dir for outputs')
    parser.add_argument("-m", dest="model_name", type=str, default='InceptionV3', help='model (default: InceptionV3)')
    parser.add_argument("-w", dest="pretrained", type=bool, default=True, help='use pretrained weights')
    parser.add_argument("-aug", dest="aug", type=int, default=0, choices=[1, 0], help='augment: 1 - ON, 0 - OFF')
    parser.add_argument("-d", dest="d", type=int, default=0, choices=[1, 0], help='debug: 1 - ON, 0 - OFF')
    parser.add_argument("-n", dest="note", type=str, default='', help='note: will be added to output file path')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.x_train_path,
                    args.y_train_path,
                    args.x_val_path,
                    args.y_val_path,
                    args.x_test_path,
                    args.output_path,
                    args.model_name,
                    args.pretrained,
                    args.aug,
                    args.d,
                    args.note)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc(file=sys.stdout)
