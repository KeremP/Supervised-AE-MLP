import torch
import torch.nn as nn
import numpy as np

class MarketDataset:
    def __init__(self, df, feat_cols, target_cols):
        self.features = df[feat_cols].values
        self.label = df[target_cols].values.reshape(-1,len(target_cols))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'targets': torch.tensor(self.label[idx], dtype=torch.float)

        }

#AE train
def train_ae(model, optimizer, dataloader ,loss_fn, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        X,y = data['features'].to(device), data['targets'].to(device)
        optimizer.zero_grad()
        outputs,_ = model(X)
        loss = loss_fn(outputs,y)
        loss.backward()
        optimizer.step()

        final_loss += loss.item()
    final_loss/=len(dataloader)

    return final_loss

#MLP train
def train_mlp(mlp, ae, optimizer, dataloder, loss_fn, device):
    mlp.train()
    ae.eval()
    final_loss = 0

    for data in dataloader:
        X,y = data['features'].to(device),data['targets'].to(device)
        optimizer.zero_grad()
        decoder,encoder = ae(X)
        outputs = mlp(X,encoder)
        loss = loss_fn(outputs,y)
        loss.backward()
        optimizer.step()

        final_loss += loss.item()
    final_loss/=len(dataloader)

    return final_loss

#MLP inference
def inference_mlp(mlp, ae, dataloader, loss_fn, device):
    mlp.eval()
    ae.eval()

    val_loss = 0
    for data in dataloader:
        X,y = data['features'].to(device),data['targets'].to(device)
        with torch.no_grad():
            decoder,encoder = ae(X)
            outputs = mlp(X,encoder)
        if loss_fn:
            loss = loss_fn(outputs,y)

            val_loss += loss.item()

    if loss_fn:
        val_loss/=len(dataloader)
    else:
        val_loss = None



    return val_loss
