"""The file contains name settings."""
import torch
file_loc = '../boxscore-data/rotowire/'
use_cuda = torch.cuda.is_available()

MAX_LENGTH = 800
MAX_SENTENCES = None
MAX_TRAIN_NUM = None

# PRETRAIN is the model name that you want read
# The naming convention is 'PRETRAIN_iterNum'
PRETRAIN = None
iterNum = None

# Default hyper-parameters for training
EMBEDDING_SIZE = 600
LR = 0.01  # Adagrad
# LR = 0.003  # Adam
EPOCH_TIME = 220
BATCH_SIZE = 2
GRAD_CLIP = 5
LAYER_DEPTH = 2

# Default parameters for display
GET_LOSS = 1
SAVE_MODEL = 5

# Choose models

# ENCODER_STYLE = 'LIN'
# ENCODER_STYLE = 'BiLSTM'
# ENCODER_STYLE = 'RNN'
# DECODER_STYLE = 'RNN'

# ENCODER_STYLE = 'HierarchicalBiLSTM'
ENCODER_STYLE = 'HierarchicalRNN'
DECODER_STYLE = 'HierarchicalRNN'
OUTPUT_FILE = 'pretrain_copy_ms5'
COPY_PLAYER = False
TOCOPY = False

# DATA PREPROCESSING
MAX_PLAYERS = 31  # information taken from rotowire
PLAYER_PADDINGS = ['<PAD' + str(i) + '>' for i in range(0, MAX_PLAYERS)]
