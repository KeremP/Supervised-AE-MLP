import torch
import torch.nn as nn
import numpy as np

#training and inference functions
#take as input: model, optimizer, list of eras, train_dataset, loss func and torch.device
#outputs: loss and predictions, respectively

#training function
def train_fn(ae, mlp, ae_opt, mlp_opt, eras, train_dataset, feat_cols, target_cols,loss_fn, device):
    ae.train()
    mlp.train()
    final_loss = 0


    for era in eras:
        df = train_dataset[train_dataset.era==era]
        X,y = torch.from_numpy(df[feat_cols].values).float().to(device),torch.from_numpy(df[target_cols].values).float().to(device)
        ae_opt.zero_grad()
        mlp_opt.zero_grad()

        #auto encoder
        ae_output, encoder = ae(X)
        # ae_loss = loss_fn(ae_output,y)
        # ae_loss.backward()

        #MLP
        output = mlp(X,encoder)
        loss = loss_fn(output,y)
        loss.backward()
        mlp_opt.step()
        ae_opt.step()

        final_loss += ae_loss.item()

    final_loss/=len(eras)

    return final_loss

#inference function - outputs val_loss and epoch predictions for eval
def inference(ae, mlp, eras, val_dataset, feat_cols, target_cols, device,loss_fn=None):
    ae.eval()
    mlp.eval()

    preds = []
    val_loss = 0
    for era in eras:
        df = val_dataset[val_dataset.era==era]
        X,y = torch.from_numpy(df[feat_cols].values).float().to(device),torch.from_numpy(df[target_cols].values).float().to(device)
        with torch.no_grad():
            _, encoder = ae(X)
            outputs = mlp(X,encoder)
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
