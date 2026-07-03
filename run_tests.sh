# Bash file for running evaluation on test data

# MoDL testing
python scripts/test_script.py --config_test=configs/modl_testing_noise_variance_0_075.yaml --config_data=configs/data.yaml
python scripts/test_script.py --config_test=configs/modl_testing_noise_variance_0_15.yaml --config_data=configs/data.yaml
python scripts/test_script.py --config_test=configs/modl_testing_noise_variance_0_30.yaml --config_data=configs/data.yaml

# spatially adaptive TV testing
python scripts/test_script.py --config_test=configs/tv_lambda_map_testing_noise_variance_0_075.yaml --config_data=configs/data.yaml
python scripts/test_script.py --config_test=configs/tv_lambda_map_testing_noise_variance_0_15.yaml --config_data=configs/data.yaml
python scripts/test_script.py --config_test=configs/tv_lambda_map_testing_noise_variance_0_30.yaml --config_data=configs/data.yaml

# convolutional synthesis dictionary with spatially adaptive sparsity maps testing
python scripts/test_script.py --config_test=configs/cdl_lambda_map_testing_noise_variance_0_075.yaml --config_data=configs/data.yaml
python scripts/test_script.py --config_test=configs/cdl_lambda_map_testing_noise_variance_0_15.yaml --config_data=configs/data.yaml
python scripts/test_script.py --config_test=configs/cdl_lambda_map_testing_noise_variance_0_30.yaml --config_data=configs/data.yaml
