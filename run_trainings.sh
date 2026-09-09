# Bash file for running trainings

# MoDL training
# python scripts/training_script.py --config_training=configs/modl_training.yaml --config_data=configs/data.yaml

# spatially adaptive TV
# python scripts/training_script.py --config_training=configs/tv_lambda_map_training.yaml --config_data=configs/data.yaml

# convolutional synthesis dictionary with spatially adaptive sparsity maps
# python scripts/training_script.py --config_training=configs/cdl_lambda_map_training.yaml --config_data=configs/data.yaml

# multi-dictionary convolutional synthesis dictionary with spatially adaptive sparsity maps
python scripts/training_script.py --config_training=configs/cdl_multi_dict_lambda_map_training.yaml --config_data=configs/data.yaml
