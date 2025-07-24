####################### Setup ########################
conda create -n fg_env python=3.9
conda activate fg_env
pip install -r requirements.txt

####################### Training #####################
python ./script/main.py dataset_dir ./datasets/Cars nb_classes 196 epochs 150 model_name srgnn

