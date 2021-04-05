import torch
import torch.nn as nn
import numpy as np

#TODO: bifourcate train functions
#TODO: refactor MarketDataset st it can handle eras as minibatches

#custom dataset
class MarketDataset:
    def __init__(self, df):
        self.features = df[feat_cols].values
        self.label = df[target_cols].values.reshape(-1,len(target_cols))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float)

        }



#training and inference functions
#take as input: model, optimizer, list of eras, train_dataset, loss func and torch.device
#outputs: loss and predictions, respectively



#AE train
def train_ae(model, optimizer, eras, train_dataset, feat_cols, target_cols, loss_fn, device):
    model.train()
    final_loss = 0

    for era in eras:
        df = train_dataset[train_dataset.era==era]
        X,y = torch.from_numpy(df[feat_cols].values).float().to(device),torch.from_numpy(df[target_cols].values).float().to(device)
        optimizer.zero_grad()
        outputs,_ = model(X)
        loss = loss_fn(outputs,y)
        loss.backward()
        optimizer.step()

        final_loss += loss.item()
    final_loss/=len(eras)

    return final_loss
#MLP train
def train_mlp(mlp, ea, optimizer, eras, train_dataset, feat_cols, target_cols, loss_fn, device):
    model.train()
    final_loss = 0

    for era in eras:
        df = train_dataset[train_dataset.era==era]
        X,y = torch.from_numpy(df[feat_cols].values).float().to(device),torch.from_numpy(df[target_cols].values).float().to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs,y)
        loss.backward()
        optimizer.step()

        final_loss += loss.item()
    final_loss/=len(eras)

    return final_loss

#inference function - outputs val_loss and epoch predictions for eval
def inference(model, eras, val_dataset, feat_cols, target_cols, device,loss_fn=None):
    model.eval()
    preds = []
    val_loss = 0
    for era in eras:
        df = val_dataset[val_dataset.era==era]
        X,y = torch.from_numpy(df[feat_cols].values).float().to(device),torch.from_numpy(df[target_cols].values).float().to(device)
        with torch.no_grad():
            outputs = model(X)
        if loss_fn:
            loss = loss_fn(outputs,y)

            val_loss += loss.item()

        preds.append(outputs.detach().cpu().numpy())



    preds = np.concatenate(preds).reshape(-1,len(target_cols))

    if loss_fn:
        val_loss/=len(eras)
    else:
        val_loss = None



    return val_loss, preds
