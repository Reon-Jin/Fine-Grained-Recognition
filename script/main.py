import sys
import os
import ast
import yaml
import json
import numpy as np
from sklearn.model_selection import train_test_split

from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention as Attention
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers import GCNConv, GlobalAttentionPool, SortPool, TopKPool, GlobalSumPool, GlobalAttnSumPool, ARMAConv, APPNPConv

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout, Flatten, LSTM
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as pp_input
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop

# user-defined functions
from RoiPoolingConvTF2 import RoiPoolingConv
from opt_dg_tf2_new import DirectoryDataGenerator
from custom_validate_callback import CustomCallback
from utils import getROIS, getIntegralROIS, crop, squeezefunc, stackfunc
from models import construct_model

tf.compat.v1.experimental.output_all_intermediates(True)

# ---------------- Load configuration ----------------
param_dir = "../config.yaml"
with open(param_dir, 'r') as file:
    param = yaml.load(file, Loader=yaml.FullLoader)
print('Loading Default parameter configuration:\n', json.dumps(param, sort_keys=True, indent=3))

# Data parameters
nb_classes = param['DATA']['nb_classes']
image_size = tuple(param['DATA']['image_size'])
dataset_dir = param['DATA']['dataset_dir']

# Hardware parameters
multi_gpu = param['HARDWARE']['multi_gpu']
gpu_id = param['HARDWARE']['gpu_id']
gpu_utilisation = param['HARDWARE']['gpu_utilisation']

# Augmentation parameters
aug_zoom = param['AUGMENTATION']['aug_zoom']
aug_tx = param['AUGMENTATION']['aug_tx']
aug_ty = param['AUGMENTATION']['aug_ty']
aug_rotation = param['AUGMENTATION']['aug_rotation']

# Model parameters
batch_size = param['MODEL']['batch_size']
lr = param['MODEL']['learning_rate']
model_name = param['MODEL']['model_name']
checkpoint_path = param['MODEL']['checkpoint']

# Training parameters
validation_freq = param['TRAIN']['validation_freq']
checkpoint_freq = param['TRAIN']['checkpoint_freq']
epochs = param['TRAIN']['epochs']

# ------------- Command-line overrides -------------
if len(sys.argv) > 2:
    total_params = len(sys.argv)
    for i in range(1, total_params, 2):
        var_name = sys.argv[i]
        new_val = sys.argv[i+1]
        try:
            exec(f"{var_name} = {new_val}")
        except:
            exec(f"{var_name} = '{new_val}'")

# ------------- GPU setup -------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('=========================== CUDA ========================', os.environ["CUDA_VISIBLE_DEVICES"])
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# Disable eager for compatibility
tf.compat.v1.disable_eager_execution()

# ------------- Additional fixed parameters -------------
lstm_units = 128
alpha = 0.3
channels = 512
completed_epochs = 0
ROIS_resolution = 42
minSize = 2
ROIS_grid_size = 3
pool_size = 7

# ------------- Directory setup -------------
working_dir = os.path.dirname(os.path.realpath(__file__))
train_data_dir = f"{dataset_dir}/train/"
# We will split train into train/validation; no separate val folder needed

# ----------------- Split train/val -----------------
# Temporary generator to list all files and labels
_temp_dg = DirectoryDataGenerator(
    base_directories=[train_data_dir],
    augmentor=False,
    preprocessors=None,
    batch_size=1,
    target_sizes=image_size,
    shuffle=False,
    verbose=False
)
all_files = _temp_dg.files
all_labels = _temp_dg.labels

# Stratified split
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=42
)
# Update sample counts
nb_train_samples = len(train_files)
nb_val_samples = len(val_files)
validation_steps = max(1, nb_val_samples // batch_size)

print(f"Training samples: {nb_train_samples}, Validation samples: {nb_val_samples}")

# ------------- Build model -------------
model = construct_model(
    name=model_name,
    pool_size=pool_size,
    ROIS_resolution=ROIS_resolution,
    ROIS_grid_size=ROIS_grid_size,
    minSize=minSize,
    alpha=alpha,
    nb_classes=nb_classes,
    batch_size=batch_size
)
# Load checkpoint if specified
if checkpoint_path and os.path.isfile(checkpoint_path):
    print("Loading weights from", checkpoint_path)
    model.load_weights(checkpoint_path)

optimizer = SGD(lr=lr)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])

# ------------- Callbacks -------------
output_model_dir = f"{working_dir}/TrainedModels/{model_name}"
metrics_dir = f"{working_dir}/Metrics/{model_name}"
csv_logger = CSVLogger(metrics_dir + f"{model_name}(Training).csv")
checkpointer = ModelCheckpoint(
    filepath=output_model_dir + '.{epoch:02d}.h5',
    verbose=1,
    save_weights_only=False,
    period=checkpoint_freq
)

# ------------- Data generators -------------
# Training generator
train_dg = DirectoryDataGenerator(
    base_directories=[train_data_dir],
    augmentor=True,
    preprocessors=pp_input,
    batch_size=batch_size,
    target_sizes=image_size,
    shuffle=True,
    verbose=1
)
# Override with split lists
train_dg.files = train_files
train_dg.labels = train_labels
train_dg.nb_files = nb_train_samples
train_dg.on_epoch_end()

# Validation generator
val_dg = DirectoryDataGenerator(
    base_directories=[train_data_dir],  # use same base_dir but files will be overridden
    augmentor=False,
    preprocessors=pp_input,
    batch_size=batch_size,
    target_sizes=image_size,
    shuffle=False,
    verbose=1
)
val_dg.files = val_files
val_dg.labels = val_labels
val_dg.nb_files = nb_val_samples
val_dg.on_epoch_end()

# ------------- Train -------------
model.fit(
    train_dg,
    steps_per_epoch=nb_train_samples // batch_size,
    initial_epoch=completed_epochs,
    epochs=epochs,
    validation_data=val_dg,
    validation_steps=validation_steps,
    callbacks=[checkpointer, csv_logger, CustomCallback(val_dg, validation_steps, metrics_dir + model_name)]
)  # train and validate the model
