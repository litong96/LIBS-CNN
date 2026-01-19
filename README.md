# LIBS-CNN
Coal geographic origins prediction using LIBS, 1D-CNN

## Models
The following models are imoplemented in this repository:

### 1. preprocessing
Preprocessing the raw LIBS data, including baseline correction:
-code:'baseline.py'
noise filter:
-code:'noise_filter.py'
and normalization:
-code:'Normalization.py'

### 2. interpretable 1D-CNN
The 1d-cnn was used with a group 5-fold cross-validation as the data partitioning strategy, and was combined with SHAP for interpretability analysis
- Code: `1dcnn_final.py`

### 3. t-SNE
tSNE is used for visualizing the model training process.
- Code: `tsne.py`
