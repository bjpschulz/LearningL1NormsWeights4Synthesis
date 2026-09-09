import mrpro
import torch
from adaptive_l1.data.utils import read_split_file
from adaptive_l1.data.augmentation import Compose, RandomFlip, RandomRotate90
from adaptive_l1.data.data_classes import LowFieldMRDataset

from adaptive_l1.models.utils import (
    define_cdl_model,
    define_cdl_multi_dict_model,
    define_tv_model,
    define_modl_model,
)
from adaptive_l1.models.utils import (
    create_cdl_run_directory,
    create_cdl_multi_dict_run_directory,
    create_tv_run_directory,
    create_modl_run_directory,
)
from adaptive_l1.models.utils import load_conv_dictionaries
from adaptive_l1.data.utils import load_config

from adaptive_l1.training.trainer import train_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_training", type=str, required=True)
parser.add_argument("--config_data", type=str, required=True)

args = parser.parse_args()

cfg_training = load_config(args.config_training)
cfg_data = load_config(args.config_data)

data_dir = cfg_data["data_dir"]
split_dir = cfg_data["split_dir"]

n_training = cfg_training["training"]["n_training"]
n_validation = cfg_training["training"]["n_validation"]
batch_size = cfg_training["training"]["batch_size"]

training_files = read_split_file(
    data_dir=data_dir,
    split_file=split_dir + "fastmri_training.txt",
)[:n_training]

validation_files = read_split_file(
    data_dir=data_dir,
    split_file=split_dir + "fastmri_validation.txt",
)[:n_validation]


training_image_data = mrpro.phantoms.FastMRIImageDataset(
    path=training_files,
    coil_combine=True,
)
validation_image_data = mrpro.phantoms.FastMRIImageDataset(
    path=validation_files,
    coil_combine=True,
)

base_seed = 42
rng = mrpro.utils.RandomGenerator(seed=base_seed)
train_transform = Compose(
    [
        RandomFlip(dim=-1, p=0.5),
        RandomFlip(dim=-2, p=0.5),
        RandomRotate90(p=0.5),
    ],
    rng,
)

noise_variance_low, noise_variance_high = (
    cfg_training["training"]["noise_variance"]["low"],
    cfg_training["training"]["noise_variance"]["high"],
)
noise_variance_dict = {"low": noise_variance_low, "high": noise_variance_high}

training_data = LowFieldMRDataset(
    image_dataset=training_image_data,
    noise_variance=noise_variance_dict,
    n_k1=cfg_training["training"]["n_k1"],
    transform=train_transform,
    base_seed=base_seed,
)
validation_data = LowFieldMRDataset(
    image_dataset=validation_image_data,
    noise_variance=noise_variance_dict,
    n_k1=cfg_training["training"]["n_k1"],
    base_seed=base_seed,
)

training_loader = torch.utils.data.DataLoader(
    training_data, batch_size=batch_size, shuffle=True
)
validation_loader = torch.utils.data.DataLoader(
    validation_data, batch_size=batch_size, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_dictionaries = None
validation_dictionaries = None


if cfg_training["model"]["name"] == "modl":
    model = define_modl_model(cfg_training)
    run_dir = create_modl_run_directory(cfg_training)

    params_list = [
        {
            "params": model.cnn_block.parameters(),
            "lr": cfg_training["training"]["learning_rate"],
            "weight_decay": cfg_training["training"]["weight_decay"],
        },
        {
            "params": model._regularization_parameter,
            "lr": cfg_training["training"]["learning_rate_scalar"],
        },
    ]

elif cfg_training["model"]["name"] == "tv":
    model = define_tv_model(cfg_training)
    run_dir = create_tv_run_directory(cfg_training)

    params_list = [
        {
            "params": model.parameter_map_network.cnn_block.parameters(),
            "lr": cfg_training["training"]["learning_rate"],
            "weight_decay": cfg_training["training"]["weight_decay"],
        },
        {
            "params": [model.parameter_map_network._global_scaling],
            "lr": cfg_training["training"]["learning_rate_global_scaling"],
        },
    ]

elif cfg_training["model"]["name"] == "cdl":
    model = define_cdl_model(cfg_training)
    run_dir = create_cdl_run_directory(cfg_training)

    params_list = [
        {
            "params": list(model.parameter_map_network.cnn_block.parameters()),
            "lr": cfg_training["training"]["learning_rate"],
            "weight_decay": cfg_training["training"]["weight_decay"],
        },
        {
            "params": [model._low_pass_filtering_parameter],
            "lr": cfg_training["training"]["learning_rate_low_pass_param"],
        },
    ]

elif cfg_training["model"]["name"] == "cdl_multi_dict":
    training_dictionaries = load_conv_dictionaries(
        cfg_data["conv_dictionary_training_dir"]
    )
    validation_dictionaries = load_conv_dictionaries(
        cfg_data["conv_dictionary_validation_dir"]
    )

    model = define_cdl_multi_dict_model(
        cfg_training, kernel=training_dictionaries[0].kernel
    )
    run_dir = create_cdl_multi_dict_run_directory(cfg_training, len(training_dictionaries))

    n_training_dictionaries = len(training_dictionaries)
    n_validation_dictionaries = len(validation_dictionaries)

    params_list = [
        {
            "params": list(model.parameter_map_network.cnn_block.parameters()),
            "lr": cfg_training["training"]["learning_rate"],
            "weight_decay": cfg_training["training"]["weight_decay"],
        },
        {
            "params": [model._low_pass_filtering_parameter],
            "lr": cfg_training["training"]["learning_rate_low_pass_param"],
        },
    ]

else:
    raise ValueError(
        f"Model name should be either 'modl', 'cdl', 'cdl_multi_dict' or 'tv',"
        f" but got {cfg_training['model']['name']}"
    )

optimizer = torch.optim.Adam(params=params_list)

# collect all hyperparameters defining the experiment
config = {
    name: value
    for name, value in locals().items()
    if isinstance(value, (int, float, tuple, str, bool))
}

model = train_model(
    model=model,
    training_loader=training_loader,
    validation_loader=validation_loader,
    optimizer=optimizer,
    loss_function=torch.nn.MSELoss(),
    device=device,
    n_epochs=cfg_training["training"]["n_epochs"],
    run_dir=run_dir,
    config=config,
    training_dictionaries=training_dictionaries,
    validation_dictionaries=validation_dictionaries,
)
