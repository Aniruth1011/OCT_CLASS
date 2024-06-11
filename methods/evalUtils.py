import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sn
import pandas as pd
from collections import Counter

def report_binary_scores(y_true_urgent, y_pred_prob_urgent, v=0):
    y_pred_urgent = y_pred_prob_urgent.round().astype(np.int)
    tn, fp, fn, tp = confusion_matrix(y_true_urgent.astype(np.float), y_pred_urgent).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    plr = tpr / fpr  
    nlr = fnr / tnr 
    acc = accuracy_score(y_true_urgent, y_pred_urgent)
    if v:
        print(f'\t accuracy: {acc:.3g}')
        print(f"\t sensitivity {tpr:.3g}")
        print(f"\t specificity {tnr:.3g}")
        print(f"\t positive likelihood ratio {plr:.3g}")
        print(f"\t negative likelihood ratio {nlr:.3g}")
    return acc, tpr, tnr, plr, nlr


def plot_model_hist(model_history, loss_name='categorical cross entropy', metric_name='acc', show=True):
    hist = model_history
    trn_loss = np.array(hist['train_loss'])
    val_loss = np.array(hist['val_loss'])
    trn_metric = np.array(hist[metric_name])
    val_metric = np.array(hist['val_' + metric_name])

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(trn_loss, c='r', label='train')
    plt.plot(val_loss, c='b', label='val')
    plt.title(f'loss ({loss_name})')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(trn_metric, c='r', label='train')
    plt.plot(val_metric, c='b', label='val')
    plt.title(metric_name)
    plt.legend()
    if show:
        plt.show()
    return fig


def one_hot_decoder(y):
    return y.argmax(axis=1)


def int2cat(y, class_map):
    return np.array([class_map[yi] for yi in y])


def get_roc(y_true, y_pred, class_map):
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 2
    y_true_1hot, y_pred_1hot = y_true, y_pred
    lbls = class_map.keys()
    tprs, fprs, aucs = {}, {}, {}
    colors = ['red', 'green', 'blue', 'orange']
    plt.figure(figsize=(10, 10))
    for lbl in lbls:
        img_class = class_map[lbl]
        fprs[lbl], tprs[lbl], _ = roc_curve(y_true[:, lbl], y_pred[:, lbl])
        aucs[lbl] = auc(fprs[lbl], tprs[lbl])
        plt.plot(fprs[lbl], tprs[lbl], color=colors[lbl], label=f'{img_class} vs rest (auc={aucs[lbl]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right")
    plt.show()

    # Urgent vs Non-Urgent
    print('Urgent vs Non-Urgent')
    class_map_r = {i: lbl for lbl, i in class_map.items()}
    y_true_urgent = urgent_v_routine(y_true_1hot, class_map_r).astype(np.int)
    y_pred_prob_urgent = urgent_v_routine(y_pred_1hot, class_map_r)
    y_pred_urgent = y_pred_prob_urgent.round().astype(np.int)
    plt.figure(figsize=(10, 10))
    fpr, tpr, _ = roc_curve(y_true_urgent, y_pred_prob_urgent)
    auc_urgent = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='r', lw=2, label=f'Urgent vs Non-Urgent (auc={auc_urgent:.3g})')
    plt.plot([0, 1], [0, 1], 'k--', label='chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right")
    plt.show()
    return aucs


def urgent_v_routine(y, class_map):
    urgent_classes = ["CNV", "DME"]
    urgent_idx = [class_map[img_class] for img_class in urgent_classes]
    return np.array([yi[urgent_idx].sum() for yi in y])


def get_confusion_matrix(y_true, y_pred, class_map, title='', plot=True):
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 2
    y_true_1hot, y_pred_1hot = y_true, y_pred
    y_true, y_pred = one_hot_decoder(y_true), one_hot_decoder(y_pred)
    acc = accuracy_score(y_true, y_pred)
    err = 1 - acc
    print(f'\t accuracy: {acc:.3g}')
    print(f'\t error: {err:.3g}')
    lbls = class_map.keys()
    cf = confusion_matrix(y_true, y_pred)
    cf_df = pd.DataFrame(cf, index=lbls, columns=lbls)
    if plot:
        sn.heatmap(cf_df, annot=True, cmap=sn.color_palette("Blues_r"))
        plt.xlabel('y Pred', fontsize=14)
        plt.ylabel('y True', fontsize=14)
        plt.title(title, fontsize=14)
        plt.show()

    for lbl in lbls:
        img_class = class_map[lbl]
        print(img_class)
        tp = cf_df.loc[lbl, lbl]
        fn = cf_df.loc[lbl][cf_df.columns != lbl].sum()
        fp = cf_df[lbl][cf_df.columns != lbl].sum()
        tn = cf_df.loc[cf_df.index != lbl, cf_df.columns != lbl].sum().sum()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        plr = tpr / fpr  # positive likelihood ratio
        nlr = fnr / tnr  # negative likelihood ratio
        print(f"\t sensitivity {tpr:.3g}")
        print(f"\t specificity {tnr:.3g}")
        print(f"\t positive likelihood ratio {plr:.3g}")
        print(f"\t negative likelihood ratio {nlr:.3g}")
        print("\n")

    # Urgent vs Non-Urgent
    print('Urgent vs Non-Urgent')
    class_map_r = {i: lbl for lbl, i in class_map.items()}
    y_true_urgent = urgent_v_routine(y_true_1hot, class_map_r).astype(np.int)
    y_pred_prob_urgent = urgent_v_routine(y_pred_1hot, class_map_r)
    y_pred_urgent = y_pred_prob_urgent.round().astype(np.int)
    tn, fp, fn, tp = confusion_matrix(y_true_urgent.round(), y_pred_urgent).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    plr = tpr / fpr  
    nlr = fnr / tnr  
    acc = accuracy_score(y_true_urgent, y_pred_urgent)
    err = 1 - acc
    print(f'\t accuracy: {acc:.3g}')
    print(f'\t error: {err:.3g}')
    print(f"\t sensitivity {tpr:.3g}")
    print(f"\t specificity {tnr:.3g}")
    print(f"\t positive likelihood ratio {plr:.3g}")
    print(f"\t negative likelihood ratio {nlr:.3g}")
    print("\n" * 6)
    return cf_df

