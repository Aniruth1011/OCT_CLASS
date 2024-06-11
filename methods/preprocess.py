from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError
import os
import traceback
import sys
from PIL import Image
import numpy as np
import pandas as pd
import skimage
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

def get_class_labels(intKey=False):
    img_type_dict = {
        "NORMAL": 0,
        "DRUSEN": 1,
        "CNV": 2,
        "DME": 3,
    }
    if intKey:
        img_type_dict = {i: l for (l, i) in img_type_dict.items()}
    return img_type_dict

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

def save_data(output_path, img_stack, target_list, new_size, dataset, n_train, img_names):
    # save as np array
    img_stack = np.stack(img_stack, axis=0)
    target_list = np.asarray(target_list)
    target_df = pd.DataFrame(index=img_names)
    target_df[dataset] = target_list

    info_tag = "{}_{}".format(str(new_size), dataset)
    if dataset == 'train':
        img_stack_out_path = os.path.join(output_path, "imgData_{}_n{}.npy".format(info_tag, n_train))
    else:
        img_stack_out_path = os.path.join(output_path, "imgData_{}.npy".format(info_tag))
    target_list_out_path = os.path.join(output_path, "targetData_{}.npy".format(info_tag))
    np.save(img_stack_out_path, img_stack)
    np.save(target_list_out_path, target_list)
    target_df.to_csv(os.path.join(output_path, "targetData_{}.csv".format(info_tag)))

def preprocess_dir(data_path, output_path, dataset, n_train, new_size):
    img_type_dict = get_class_labels()
    print('Preprocessing:', dataset)
    transform = transforms.Compose([
        transforms.Resize(new_size[:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = NestedFolderDataset(data_path, img_type_dict, transform=transform)
    dataloader = DataLoader(dataset, batch_size=n_train, shuffle=True)
    
    img_stack, target_list, img_names = [], [], []
    for i, (img_batch, label_batch) in enumerate(dataloader):
        img_stack.append(img_batch)
        target_list.append(label_batch)
        img_names += [f"{i}_{j}" for j in range(img_batch.size(0))]
    
    img_stack = torch.cat(img_stack).numpy()
    target_list = torch.cat(target_list).numpy()
    save_data(output_path, img_stack, target_list, new_size, dataset, n_train, img_names)

def get_parser():
    """defines the parser for this script"""
    module_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-i", dest="data_path", type=str, help="the location dataset")
    module_parser.add_argument("-o", dest="output_path", type=str, help='base dir for outputs')
    module_parser.add_argument("-subdir", dest="subdir", type=str, choices=['test', 'train', 'val', 'all'], help='subdir: trn, test, val, or all ...')
    module_parser.add_argument("-n", dest="n_train", type=int, help='n: number of images for training')
    module_parser.add_argument("-Rx", dest="x_res", type=int, help='x resulution for final img')
    module_parser.add_argument("-Ry", dest="y_res", type=int, help='y resolution of final image')
    module_parser.add_argument("-d", dest="d", type=int, default=0, help='debug')
    return module_parser

def main_driver(data_path, output_path, subdir, n_train, x_res, y_res, d):
    if d == 1:
        print('debug mode: ON')
        subdir = 'train'
        n_train = 10

    assert (os.path.isdir(data_path))
    new_size = (int(x_res), int(y_res), 3)
    if not (os.path.isdir(output_path)):
        os.makedirs(output_path)
    print(output_path)
    if subdir == 'all':
        for subdir in ['test', 'train', 'val']:
            preprocess_dir(os.path.join(data_path, subdir), output_path, subdir, n_train, new_size)
    else:
        preprocess_dir(os.path.join(data_path, subdir), output_path, subdir, n_train, new_size)

if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.data_path, args.output_path, args.subdir, args.n_train, args.x_res, args.y_res, args.d)
        print('Done!')
    except ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
