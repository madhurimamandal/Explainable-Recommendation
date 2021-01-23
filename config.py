import torch
USE_CUDA = torch.cuda.is_available()
MAX_LENGTH = 100
teacher_forcing_ratio = 1.0
save_dir = 'data'
factor_num=16
num_layers=4
dropout=0.0
model = 'NeuMF-end'