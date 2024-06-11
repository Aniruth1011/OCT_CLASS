import os
import sys
import traceback
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
from methods.preprocess import get_class_labels


class NestedFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.img_labels = []
        
        for class_label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_label)
            if os.path.isdir(class_dir):
                for subfolder in os.listdir(class_dir):
                    subfolder_dir = os.path.join(class_dir, subfolder)
                    if os.path.isdir(subfolder_dir):
                        for img_file in os.listdir(subfolder_dir):
                            img_path = os.path.join(subfolder_dir, img_file)
                            if img_path.lower().endswith('.tiff'):
                                self.img_labels.append((img_path, int(class_label)))
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_binary_pred(model_pred_df):
    urgent_labels = ['CNV', 'DME']
    urgent_cols = []
    pred_urgent_df = pd.DataFrame(index=model_pred_df.index)
    for col in model_pred_df.columns:
        if (urgent_labels[0] in col) or (urgent_labels[1] in col):
            urgent_cols.append(col)
    assert (len(urgent_cols) == 2)
    pred_urgent_df['urgent_proba'] = model_pred_df[urgent_cols[0]] + model_pred_df[urgent_cols[1]]
    return pred_urgent_df

def mean_prediction(model_pred_df, y_vals=[0, 1, 2, 3]):
    img_type_dict = get_class_labels(intKey=True)
    mean_pred = pd.DataFrame(index=model_pred_df.index)
    for yi in np.unique(y_vals):
        mean = model_pred_df.filter(regex='_{}'.format(yi)).mean(axis=1)
        mean_pred['proba_{}'.format(img_type_dict[yi])] = mean
    mean_pred = mean_pred.div(mean_pred.sum(axis=1), axis=0)
    return mean_pred

def load_model(model_name):
    if model_name == 'InceptionV3':
        model = models.inception_v3(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
    elif model_name == 'VGG16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 4)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
    elif model_name == 'Xception':
        raise ValueError('Xception is not available in torchvision. Use another model.')
    else:
        raise ValueError('Unknown model name')
    return model

def preprocess_imgs(x_path, new_size):
    transform = transforms.Compose([
        transforms.Resize((new_size[0], new_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = NestedFolderDataset(x_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader, dataset.img_files

def save_model_results(model_name, y_pred, img_names, model_dir_path, img_type_dict):
    cols = ["{}_{}_{}".format(model_name, i, l) for (l, i) in img_type_dict.items()]
    y_pred_df = pd.DataFrame(y_pred, columns=cols, index=img_names)
    y_pred_df.to_csv(os.path.join(model_dir_path, "{}_predictions.csv".format(model_name)))
    return y_pred_df

def save_predictions(model_pred_dict, models, img_names, out_path):
    model_pred_df = pd.DataFrame(index=img_names)
    for modelName in models:
        model_pred_df = pd.merge(model_pred_df, model_pred_dict[modelName], left_index=True, right_index=True)
    mean_pred_df = mean_prediction(model_pred_df)
    binary_pred_df = get_binary_pred(mean_pred_df)
    model_pred_df.to_csv(os.path.join(out_path, "individualModelPredictions.csv"))
    mean_pred_df.to_csv(os.path.join(out_path, "ensembleClfMeanProba.csv"))
    binary_pred_df.to_csv(os.path.join(out_path, "urgentProba.csv"))

def get_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", dest="x_path", type=str, help="input image directory")
    parser.add_argument("-o", dest="out_path", type=str, help="output dir path")
    parser.add_argument("-m", dest="ensemble_path", type=str, help="path to model directory")
    return parser

def main(x_path, out_path, ensemble_path):
    img_type_dict = get_class_labels(intKey=True)
    model_pred_dict = {}
    models = ['InceptionV3', 'VGG16', 'ResNet50']  

    for model_name in models:
        model = load_model(model_name)
        model.eval()
        data_loader, img_names = preprocess_imgs(x_path, (299, 299, 3) if model_name != "VGG16" else (224, 224, 3))
        
        y_pred = []
        with torch.no_grad():
            for images, _ in data_loader:
                outputs = model(images)
                y_pred.append(outputs.numpy())
        
        y_pred = np.concatenate(y_pred, axis=0)
        model_dir_path = os.path.join(ensemble_path, model_name)
        os.makedirs(model_dir_path, exist_ok=True)
        y_pred_df = save_model_results(model_name, y_pred, img_names, model_dir_path, img_type_dict)
        model_pred_dict[model_name] = y_pred_df
    
    save_predictions(model_pred_dict, models, img_names, out_path)

if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main(args.x_path, args.out_path, args.ensemble_path)
        print('predict.py ... done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
