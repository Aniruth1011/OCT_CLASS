import os
import sys
import traceback
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from collections import Counter

# Custom imports based on your original script
# Assuming you have similar utilities in PyTorch
from evalUtils import UrgentVRoutne, reportBinaryScores
from trainCNN import loadTargetData, getPreprocess
from preprocess import getClassLabels

def get_parser():
    """Defines the parser for this script"""
    module_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    module_parser.add_argument("-x", dest="x_test_path", type=str,
                               help='x test path')
    module_parser.add_argument("-y", dest="y_test_path", type=str,
                               default=None,
                               help='model weights')
    module_parser.add_argument("-m", dest="model_path", type=str,
                               help='model path')
    module_parser.add_argument("-n", dest="note",
                               type=str,
                               default='',
                               help='note: will be added to output file path')
    return module_parser

def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_pred = model(X_test_tensor).numpy()
    return y_test_pred

def save_pred(model_name, output_path, y_test, y_test_pred, y_idx, note):
    y_test_pred_path = os.path.join(output_path, 'yPred_{}.npy'.format(note))
    model_pred_df = pd.DataFrame(index=y_idx)
    for y_lbl in np.unique(y_test):
        model_pred_df[model_name + "_{}".format(y_lbl)] = y_test_pred[:, y_lbl]
    model_pred_df['yTrueTest'] = y_test
    model_pred_df.to_csv(os.path.join(output_path,
                                    'yPredDf_{}.csv'.format(note)))
    np.save(y_test_pred_path, y_test_pred)

def eval_test_set(y_test, y_test_pred, model_name, output_path, note):
    class_map = getClassLabels()
    y_true_1hot = F.one_hot(torch.tensor(y_test), num_classes=len(class_map)).numpy()
    y_true_test_urgent = UrgentVRoutne(y_true_1hot, class_map).astype(np.int)
    class_acc = accuracy_score(y_test, y_test_pred.argmax(axis=1))
    print('\t accuracy: {0:.3g}'.format(class_acc))
    y_test_pred_urgent = UrgentVRoutne(y_test_pred, class_map)
    print('\t binary (urgent vs non-urgent)')
    scores = reportBinaryScores(y_true_test_urgent, y_test_pred_urgent, v=1)
    acc, tpr, tnr, plr, nlr = scores
    fprs, tprs, _ = roc_curve(y_true_test_urgent, y_test_pred_urgent)
    auc_urgent = auc(fprs, tprs)
    with open(os.path.join(output_path, 'eval_{}.txt'.format(note)), 'w') as fid:
        fid.write("model: {} \n".format(model_name))
        fid.write("4 classAcc: {} \n".format(class_acc))
        fid.write("{} \n".format('binary - urgent vs non-urgent'))
        fid.write("acc: {} \n".format(acc))
        fid.write("tpr: {} \n".format(tpr))
        fid.write("tnr: {} \n".format(tnr))
        fid.write("auc_urgent: {} \n".format(auc_urgent))

def main_driver(X_test_path,
                y_test_path,
                model_path,
                note):
    """
    Run inference on new data
    :param X_test_path: path to preprocessed image data npy array (str)
    :param y_test_path: path to image labels file [npy arr .npy or pd dataframe .csv]
    :param model_path: path to model file (str)
    :param note: additional note for output (str)
    :return: None
    """
    print('test path:', X_test_path)

    """############################################################################
                        0. Load Data
    ############################################################################"""

    assert(os.path.isfile(X_test_path))
    assert(os.path.isfile(model_path))
    if y_test_path is not None:
        assert(os.path.isfile(y_test_path))
        y_test, y_idx = loadTargetData(y_test_path)
    X_test = np.load(X_test_path)
    output_path = os.path.dirname(model_path)

    """############################################################################
                        1. Preprocess Data
    ############################################################################"""
    models = ['InceptionV3', 'VGG16', 'ResNet50', 'Xception']
    idx_list = [model in model_path for model in models]
    assert(sum(idx_list) == 1)
    model_name = models[np.argmax(idx_list)]
    preprocess_input = getPreprocess(model_name)
    X_test = preprocess_input(X_test)

    """############################################################################
                        2. Load Model and Run Inference
    ############################################################################"""
    model = torch.load(model_path)
    y_test_pred = predict(model, X_test)
    save_pred(model_name, output_path, y_test, y_test_pred, y_idx, note)

    """############################################################################
                        3. Evaluate Inference (optional)
    ############################################################################"""
    if y_test is not None:
        eval_test_set(y_test, y_test_pred, model_name, output_path, note)

if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
        main_driver(args.x_test_path,
                    args.y_test_path,
                    args.model_path,
                    args.note)
        print('Done!')

    except argparse.ArgumentError as arg_exception:
        traceback.print_exc()
    except Exception as exception:
        traceback.print_exc()
    sys.exit()
