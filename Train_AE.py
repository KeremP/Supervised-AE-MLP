import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn import cluster
from tqdm import tqdm, trange

from earlystopping import EarlyStopping
from purgedSplit import PurgedGroupTimeSeriesSplit
from utils import train_ae, inference

from ae_mlp import AE, MLP
device = torch.device('cuda')

data_path = "../Numerai/numerai_dataset_258/numerai_training_data.csv"

print("Reading data...")
train = pd.read_csv(data_path)

print("Optimizing memory usage...")
#Dataset is large, quick and dirty memory optimization
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns})

feat_cols = [c for c in train.columns if "feature" in c]

#scaling
train[feat_cols] = train[feat_cols] - 0.5

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
hidden_units = [96, 96, 896, 448, 448, 256]
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

    #optimizers
    ae_opt = torch.optim.Adam(auto_encoder.parameters(),lr=lr)

    #LR schedulers
    ae_scheduler = torch.optim.lr_scheduler.CyclicLR(ae_opt,base_lr=lr,max_lr=3e-2,cycle_momentum=False)

    #MSE loss
    loss_fn = nn.MSELoss()

    #Earlystopping
    model_weights = f'{CACHE_PATH}/model_{_fold}.pkl'
    es = EarlyStopping(patience=10,mode='min')

    #splitting dataset
    #shuffling eras - feeding eras as mini-batches
    train_dataset = train_set.loc[tr]
    valid_dataset = train_set.loc[te]
    train_eras = train_dataset.era.unique()
    valid_eras = valid_dataset.era.unique()
    # np.random.shuffle(train_eras)
    # np.random.shuffle(valid_eras)


    for epoch in (t:=trange(EPOCHS)):
        train_loss = train_ae(auto_encoder, ae_opt, train_eras, train_dataset, feat_cols, 'target', loss_fn, device)
        ae_scheduler.step()

        valid_loss, valid_preds = inference(auto_encoder, valid_eras, valid_dataset, feat_cols, 'target', device, loss_fn)
        nn.utils.clip_grad_norm_(auto_encoder.parameters(),5)

        es(valid_loss,model,model_path=model_weights)
        if es.early_stop:
            print('Early stopping')
            break

        t.set_description('Train loss {} Valid loss {}'.format(train_loss,valid_loss))
