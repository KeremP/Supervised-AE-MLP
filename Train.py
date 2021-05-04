import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import cluster
from tqdm import tqdm, trange

from earlystopping import EarlyStopping
from purgedSplit import PurgedGroupTimeSeriesSplit
from utils import train_ae, train_mlp, inference_mlp, MarketDataset

from ae_mlp import AE, MLP
device = torch.device('cuda')

data_path = "../Numerai/numerai_dataset_258/numerai_training_data.csv"

print("Reading data...")
train = pd.read_csv(data_path)

print("Optimizing memory usage...")
#Dataset is large, quick and dirty memory optimization
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns})

feat_cols = [c for c in train.columns if "feature" in c]
target_cols = ["target"]

train['erano'] = train.era.str.slice(3).astype(int)

#hold out sets
train_set = train.query('erano < 100').reset_index(drop=True)


#PurgedGroupTimeSeriesSplit - K-fold CV, prevents leakage of data from train to validation
print('PurgedGroupTimeSeriesSplit')
gkf = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=20)
#five splits, grouped by eras, gap of 20.
splits = list(gkf.split(train_set['target'],groups=train_set['erano'].values))


#params
num_cols = len(feat_cols)
num_labels = 1
hidden_units = [512, 256, 128, 96]
drop_rates = [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448]
lr = 1e-3
CACHE_PATH = './models'
EPOCHS = 200

#training loop
for _fold, (tr,te) in enumerate(splits):
    print(f'Fold: {_fold}')
    # seed_everything(seed=1111+_fold)

    #define models, push to CUDA device
    auto_encoder = AE(num_cols,num_labels,hidden_units,drop_rates)
    auto_encoder = auto_encoder.to(device)
    mlp = MLP(num_cols, hidden_units[-1],1,hidden_units,drop_rates).to(device)

    #optimizers
    ae_opt = torch.optim.Adam(auto_encoder.parameters(),lr=lr)
    mlp_opt = torch.optim.Adam(mlp.parameters(),lr=lr)

    #LR schedulers
    ae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ae_opt,'min')
    mlp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mlp_opt,'min')

    #MSE loss
    loss_fnAE = nn.MSELoss()
    loss_fnMLP = nn.MSELoss()

    #Earlystopping
    model_weights = f'{CACHE_PATH}/model_{_fold}.pkl'
    es = EarlyStopping(patience=10,mode='min')

    #train-test split
    trainDataset = MarketDataset(train.loc[tr],feat_cols,target_cols)
    valDataset = MarketDataset(train.loc[te],feat_cols,target_cols)

    trainLoader = DataLoader(trainDataset,batch_size=128)
    valLoader = DataLoader(trainDataset,batch_size=128)


    for epoch in (t:=trange(EPOCHS)):
        train_lossAE = train_ae(auto_encoder, ae_opt, trainLoader, loss_fnAE, device)
        ae_scheduler.step(train_lossAE)

        train_lossMLP = train_mlp(mlp,auto_encoder,mlp_opt,trainLoader,loss_fnMLP,device)
        valid_loss, valid_preds = inference_mlp(mlp, auto_encoder, valLoader, device, loss_fnMLP)
        mlp_scheduler(valid_loss)
        nn.utils.clip_grad_norm_(auto_encoder.parameters(),5)

        es(valid_loss,auto_encoder,model_path=model_weights)
        if es.early_stop:
            print('Early stopping')
            break

        t.set_description('AutoEncoder loss {} MLP Loss {} Valid loss {}'.format(train_lossAE,train_lossMLP,valid_loss))
