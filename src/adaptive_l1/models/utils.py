import torch 
import re
from dataclasses import dataclass
from adaptive_l1.models.spatially_adaptive_conv_synthesis import (
    SpatiallyAdaptiveConvSynthesisNet2D,
    ConvSynthesisParameterMapNetwork2D,
    FilterwiseConvSynthesisParameterMapNetwork2D,
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


@dataclass
class ConvDictionary:
    """A convolutional dictionary given by its name and kernel tensor."""

    name: str
    kernel: torch.Tensor


def load_conv_dictionaries(directory: str | Path) -> list[ConvDictionary]:
    """Load a set of pre-trained convolutional dictionary kernel tensors from a directory.

    The files are expected to follow the naming convention
    ``K{n_filters}_k{ky}x{kx}_sparsity_param{...}_lowpass_param{...}.pt``, with kernel
    tensors of shape ``(n_filters, ky, kx)`` or ``(n_filters, 1, ky, kx)``.

    Parameters
    ----------
    directory
        directory containing the convolutional dictionary filters as .pt files.
    """

    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Dictionary directory does not exist: {directory}")

    dictionaries = []
    for dict_file in sorted(directory.glob("*.pt")):
        match = re.fullmatch(
            r"K(\d+)_k(\d+)x(\d+)_sparsity_param.+_lowpass_param.+",
            dict_file.stem,
        )
        if match is None:
            print(f"Warning: Could not parse dictionary file name {dict_file.name}, skipping")
            continue

        n_filters, ky, kx = (int(group) for group in match.groups())
        kernel = torch.load(dict_file, map_location="cpu")

        expected_shapes = ((n_filters, ky, kx), (n_filters, 1, ky, kx))
        if kernel.shape not in expected_shapes:
            print(
                f"Warning: Match fail for {dict_file.stem}, got {kernel.shape},"
                f" expected one of {expected_shapes}"
            )
            continue

        dictionaries.append(ConvDictionary(name=dict_file.stem, kernel=kernel))

    if not dictionaries:
        raise ValueError(f"No valid dictionary files found in {directory}")

    print(f"* Loaded {len(dictionaries)} convolutional dictionaries from {directory}")

    return dictionaries


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
    kernel_size = cfg["model"]["conv_dictionary"]["n_conv_kernel_size"]
    
    hyperparameters_identification = f"n_enc_stages{n_enc_stages}_"\
            f"n_convs_per_stage{n_convs_per_stage}_n_filters{n_filters}_"\
            f"n_conv_dictionary_filters{n_conv_dictionary_filters}_conv_kernel_size{kernel_size}"
            
    run_dir_name = Path(f"{cfg["model"]["name"]}")
    run_dir = Path("runs") / run_dir_name  / hyperparameters_identification
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def define_cdl_multi_dict_model(cfg, kernel: torch.Tensor):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_ch_in = cfg["model"]["cnn_block"]["n_ch_in"]
    n_ch_out = cfg["model"]["cnn_block"]["n_ch_out"]
    if n_ch_in != 2 or n_ch_out != 1:
        raise ValueError(
            "The cnn_block of the multi-dictionary cdl model must be a 2-to-1 channel"
            f" U-Net, but got n_ch_in={n_ch_in} and n_ch_out={n_ch_out}"
        )

    n_enc_stages = cfg["model"]["cnn_block"]["n_enc_stages"]
    n_convs_per_stage = cfg["model"]["cnn_block"]["n_convs_per_stage"]
    n_filters = cfg["model"]["cnn_block"]["n_filters"]
    cnn_block = UNet(
        dim=2,
        n_ch_in=n_ch_in,
        n_ch_out=n_ch_out,
        n_enc_stages=n_enc_stages,
        n_convs_per_stage=n_convs_per_stage,
        n_filters=n_filters,
        kernel_size=3,
        pooling_kernel_size=2,
        bias=False,
    )

    parameter_map_network = FilterwiseConvSynthesisParameterMapNetwork2D(
        cnn_block = cnn_block,
        upper_bound = cfg["model"]["parameter_map_network"]["upper_bound"],
        sigmoid_beta = cfg["model"]["parameter_map_network"]["sigmoid_beta"],
    )
    model = SpatiallyAdaptiveConvSynthesisNet2D(
        kernel = kernel,
        parameter_map_network=parameter_map_network,
        n_iterations=cfg["model"]["n_iterations"]
    ).to(device)

    return model

def create_cdl_multi_dict_run_directory(cfg, n_training_dictionaries: int):

    n_enc_stages = cfg["model"]["cnn_block"]["n_enc_stages"]
    n_convs_per_stage = cfg["model"]["cnn_block"]["n_convs_per_stage"]
    n_filters = cfg["model"]["cnn_block"]["n_filters"]

    hyperparameters_identification = f"n_enc_stages{n_enc_stages}_"\
            f"n_convs_per_stage{n_convs_per_stage}_n_filters{n_filters}_"\
            f"n_dictionaries{n_training_dictionaries}"

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