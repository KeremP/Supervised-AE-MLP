import os
import pandas as pd
from mlfinlab.clustering import onc
import numpy as np
from mlfinlab.codependence import get_dependence_matrix

data_path = "../Numerai/numerai_dataset_257/numerai_training_data.csv"

#Read dataset
train = pd.read_csv(data_path)

print("Optimizing memory usage...")
#Dataset is large, quick and dirty memory optimization
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns})

feat_cols = [c for c in train.columns if "feature" in c]
train['erano'] = train.era.str.slice(3).astype(int)

#TODO: groupby eras...

print("Calculating era-wise corr matrix")
#Calc corr matrix
for era in train[erano].values:
    print(f"era {era}")
    corr = train[feat_cols].corr()

    print("finding optimal number of clusters")
    #ONC clusters
    corr_matrix, clusters, silhscores = onc.get_onc_clusters(corr)

    print(clusters)
