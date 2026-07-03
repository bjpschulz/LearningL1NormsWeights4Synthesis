# script for re-creating the lambda maps (Figure 2) in our EUSIPCO paper for a pre-trained CDL model, 
# showing the different sparsity level maps and their associated filters
python scripts/sparsity_level_maps_evaluation.py \
    --config_data=configs/data.yaml \
    --config_test=configs/cdl_lambda_map_testing_noise_variance_0_30.yaml 
    
# script showcasing the improvement from the best scalar sparsity parameter (obtained by line-search)
# compared to the learned spatial sparsity level maps
python scripts/scalar_sparsity_vs_spatially_adaptive_sparsity.py \
    --config_data=configs/data.yaml \
    --config_test=configs/cdl_lambda_map_testing_noise_variance_0_30.yaml 


# script for comparing TV-Lambda, CDL-Lambda and MoDL on an example of the fastMRI data
python scripts/models_comparison.py \
    --config_data=configs/data.yaml \
    --config_cdl=configs/cdl_lambda_map_testing_noise_variance_0_30.yaml \
    --config_tv=configs/tv_lambda_map_testing_noise_variance_0_30.yaml \
    --config_modl=configs/modl_testing_noise_variance_0_30.yaml     

# script for visualizing a comparison between adjoint reconstruction, TV-Lambda, CDL-Lambda and MoDL
python scripts/test_invivo_comparison.py  \
    --config_data=configs/data.yaml \
    --config_cdl=configs/cdl_lambda_map_testing_noise_variance_0_30.yaml \
    --config_tv=configs/tv_lambda_map_testing_noise_variance_0_30.yaml \
    --config_modl=configs/modl_testing_noise_variance_0_30.yaml     

