# Supervised Auto-Encoder MLP

Pytorch implementation of a Supervised Auto-Encoder MLP model for use in finanical ML competitions.

Idea is that AE will be trained to generate a reduced dimension (encoder output) representation of dataset, then train MLP on concatenation of encoder output and raw input.

Loss function can be modified wrt task (e.g. BCE for classification, MSE for regression, etc.). May require modifying output activation functions (e.g. from ReLU to Sigmoid for binary trade classification).

Some code is recycled from my Resnet repo (e.g. GaussNoise layer, earlystopping, purged K-fold).

Based on: https://www.kaggle.com/gogo827jz/jane-street-supervised-autoencoder-mlp
