# Real-Time-Network-Latency-Estimation-with-Pre-Trained-Generative-Models
Source code for paper: 
L. Deng et al. Real-Time Network Latency Estimation with Pre-Trained Generative Models, IEEE TNNLS, 2025.

Requiments:

torch==1.2.0+cu92

tqdm==4.64.1

script==1.7.3

numpy==1.21.6

Runing example using the Seattle dataset:
1, Run "Seattle_train_GMC_svd.py" to pre-train the generative model.
2, and then run "Seattle_nn_GMC_svd.py" to train the optimizer
3, finally run "Seattle_test_GMC_svd.py" for inference
