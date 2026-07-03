import torch 
from adaptive_l1.models.spatially_adaptive_conv_synthesis import (
    SpatiallyAdaptiveConvSynthesisNet2D,
    ConvSynthesisParameterMapNetwork2D,
)
from adaptive_l1.models.modl import MoDLBlock, MoDL
from adaptive_l1.models.spatially_adaptive_tv import (
    SpatiallyAdaptiveTVNet2D,
    TVParameterMapNetwork2D,
)
from adaptive_l1.models.unet import UNet

from adaptive_l1.data.utils import load_config

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CFG_DATA_PATH = PROJECT_ROOT / "configs" / "data.yaml"
cfg_data = load_config(CFG_DATA_PATH)

def define_cdl_model(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_enc_stages = cfg["model"]["cnn_block"]["n_enc_stages"]
    n_convs_per_stage = cfg["model"]["cnn_block"]["n_convs_per_stage"]
    n_filters = cfg["model"]["cnn_block"]["n_filters"]
    cnn_block = UNet(
        dim=2,
        n_ch_in=cfg["model"]["cnn_block"]["n_ch_in"],
        n_ch_out=cfg["model"]["cnn_block"]["n_ch_out"],
        n_enc_stages=n_enc_stages,
        n_convs_per_stage=n_convs_per_stage,
        n_filters=n_filters,
        kernel_size=3,
        pooling_kernel_size=2,
        bias=False,
    )

    n_conv_dictionary_filters = cfg["model"]["conv_dictionary"]["n_conv_kernel_filters"]
    kernel_size = cfg["model"]["conv_dictionary"]["n_conv_kernel_size"]
    sparsity_param = cfg["model"]["conv_dictionary"]["sparsity_param"]
    lowpass_param = cfg["model"]["conv_dictionary"]["lowpass_param"]

    conv_dictionary_kernel_dir = cfg_data["conv_dictionary_dir"]
    kernel_fname = Path(f"K{n_conv_dictionary_filters}_k{kernel_size}x{kernel_size}_"\
                    f"sparsity_param{str(sparsity_param).replace(".","_")}_"\
                    f"lowpass_param{str(lowpass_param).replace(".","_")}.pt")
    conv_dictionary_kernel = torch.load(Path(conv_dictionary_kernel_dir) / kernel_fname)

    parameter_map_network = ConvSynthesisParameterMapNetwork2D(cnn_block = cnn_block)
    model = SpatiallyAdaptiveConvSynthesisNet2D(
        kernel = conv_dictionary_kernel,
        parameter_map_network=parameter_map_network,
        n_iterations=cfg["model"]["n_iterations"]
    ).to(device)

    return model

def create_cdl_run_directory(cfg):
    
    n_enc_stages = cfg["model"]["cnn_block"]["n_enc_stages"]
    n_convs_per_stage = cfg["model"]["cnn_block"]["n_convs_per_stage"]
    n_filters = cfg["model"]["cnn_block"]["n_filters"]
    
    n_conv_dictionary_filters = cfg["model"]["conv_dictionary"]["n_conv_kernel_filters"]
    
    n_conv_dictionary_filters = cfg["model"]["conv_dictionary"]["n_conv_kernel_filters"]
    kernel_size = cfg["model"]["conv_dictionary"]["n_conv_kernel_size"]
    
    hyperparameters_identification = f"n_enc_stages{n_enc_stages}_"\
            f"n_convs_per_stage{n_convs_per_stage}_n_filters{n_filters}_"\
            f"n_conv_dictionary_filters{n_conv_dictionary_filters}_conv_kernel_size{kernel_size}"
            
    run_dir_name = Path(f"{cfg["model"]["name"]}")
    run_dir = Path("runs") / run_dir_name  / hyperparameters_identification
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def define_tv_model(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_enc_stages = cfg["model"]["cnn_block"]["n_enc_stages"]
    n_convs_per_stage = cfg["model"]["cnn_block"]["n_convs_per_stage"]
    n_filters = cfg["model"]["cnn_block"]["n_filters"]
    cnn_block = UNet(
        dim=2,
        n_ch_in=cfg["model"]["cnn_block"]["n_ch_in"],
        n_ch_out=cfg["model"]["cnn_block"]["n_ch_out"],
        n_enc_stages=n_enc_stages,
        n_convs_per_stage=n_convs_per_stage,
        n_filters=n_filters,
        kernel_size=3,
        pooling_kernel_size=2,
        bias=False,
    )
    
    parameter_map_network = TVParameterMapNetwork2D(cnn_block = cnn_block)
    model = SpatiallyAdaptiveTVNet2D(
        parameter_map_network=parameter_map_network, 
        n_iterations=cfg["model"]["n_iterations"]
    ).to(device)

    return model

def create_tv_run_directory(cfg):
    
    n_enc_stages = cfg["model"]["cnn_block"]["n_enc_stages"]
    n_convs_per_stage = cfg["model"]["cnn_block"]["n_convs_per_stage"]
    n_filters = cfg["model"]["cnn_block"]["n_filters"]
    
    hyperparameters_identification = f"n_enc_stages{n_enc_stages}_"\
            f"n_convs_per_stage{n_convs_per_stage}_n_filters{n_filters}"
    
    run_dir_name = Path(f"{cfg["model"]["name"]}")
    run_dir = Path("runs") / run_dir_name  / hyperparameters_identification
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def define_modl_model(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_layers = cfg["model"]["cnn_block"]["n_layers"]
    n_filters = cfg["model"]["cnn_block"]["n_filters"]
    cnn_block = MoDLBlock(n_layers=n_layers,
                          n_ch_in=cfg["model"]["cnn_block"]["n_ch_in"],
                          n_ch_out=cfg["model"]["cnn_block"]["n_ch_out"],
                          n_filters=n_filters
                          )
    model = MoDL(cnn_block=cnn_block, n_iterations=cfg["model"]["n_iterations"]).to(device)
    
    return model

def create_modl_run_directory(cfg):
    
    n_layers = cfg["model"]["cnn_block"]["n_layers"]
    n_filters = cfg["model"]["cnn_block"]["n_filters"]
    
    hyperparameters_identification = f"n_layers{n_layers}_n_filters{n_filters}"
    run_dir_name = Path(f"{cfg["model"]["name"]}")
    run_dir = Path("runs") / run_dir_name  / hyperparameters_identification
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir