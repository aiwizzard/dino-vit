import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def compute_knn(backbone, data_loader_train, data_loader_val):
    r""" Perfomr KNN classifier on cls embedding of 
    the visionTrasnformer.
    """
    device = next(backbone.parameters()).device
    
    data_loaders = {
        "train": data_loader_train,
        "val": data_loader_val,
    }

    data_items = {
        "X_train": [],
        "y_train": [],
        "X_val": [],
        "y_val": []
    }

    for name, data_loader in data_loaders.items():
        for imgs, y in data_loader:
            imgs = imgs.to(device)
            data_items[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            data_items[f"y_{name}"].append(y.detach().cpu().numpy())
    
    data = {k: np.concatenate(l) for k, l in data_items.items()}
    
    estimator = KNeighborsClassifier()
    estimator.fit(data["X_train"], data["y_train"])
    y_val_pred = estimator.predict(data["X_val"])

    acc = accuracy_score(data["y_val"], y_val_pred)
    return acc

def compute_embedding(backbone, data_loader):
    r"""Compute CLS embedding and prepare for Tensorboard
    """
    device = next(backbone.parameters()).device

    embs_list = []
    imgs_list = []
    labels = []

    for img, y in data_loader:
        img = img.to(device)
        embs_list.append(backbone(img).detach().cpu())
        imgs_list.append(((img * 0.224) + 0.45).cpu())
        labels.extend([data_loader.dataset.classes[i] for i in y.tolist()])

    embs = torch.cat(embs_list, dim=0)
    imgs = torch.cat(imgs_list, dim=0)
    
    return embs, imgs, labels
