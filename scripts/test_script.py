import mrpro
import torch
from adaptive_l1.data.utils import read_split_file
from adaptive_l1.data.data_classes import LowFieldMRDataset

from adaptive_l1.models.utils import define_cdl_model, define_tv_model, define_modl_model
from adaptive_l1.models.utils import create_cdl_run_directory, create_tv_run_directory, create_modl_run_directory

from adaptive_l1.data.utils import load_config

from adaptive_l1.testing.tester import test_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_test", type=str, required=True)
parser.add_argument("--config_data", type=str, required=True)

args = parser.parse_args()

cfg_test = load_config(args.config_test)
cfg_data = load_config(args.config_data)

noise_variance = cfg_test["testing"]["noise_variance"]
if not (isinstance(noise_variance, float) or isinstance(noise_variance, int)):
    raise ValueError(f"noise standard deviation should be float or integer, got {noise_variance}")

n_k1 = cfg_test["testing"]["n_k1"]
if not (isinstance(n_k1, float) or isinstance(n_k1, int)):
    raise ValueError(f"A single number of n_k1 samples (`float` or `int`) should be "
                     f"given for model testing; got {type(n_k1)}")

data_dir = cfg_data["data_dir"]
split_dir = cfg_data["split_dir"]

batch_size = cfg_test["testing"]["batch_size"]

n_test = cfg_test["testing"]["n_test"]
test_files = read_split_file(
    data_dir=data_dir,
    split_file=split_dir + "fastmri_test.txt",
)[:n_test]

test_image_data = mrpro.phantoms.FastMRIImageDataset(
    path=test_files,
    coil_combine=True,
)

base_seed = 42

test_data = LowFieldMRDataset(
    image_dataset=test_image_data,
    noise_variance=noise_variance,
    n_k1=n_k1,
    base_seed=base_seed,
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg_test["model"]["name"] == "modl":
    model = define_modl_model(cfg_test)
    run_dir = create_modl_run_directory(cfg_test)
    
elif cfg_test["model"]["name"] == "cdl":
    model = define_cdl_model(cfg_test)
    run_dir = create_cdl_run_directory(cfg_test)
    
elif cfg_test["model"]["name"] == "tv":
    model = define_tv_model(cfg_test)
    run_dir = create_tv_run_directory(cfg_test)
    
else:
    raise ValueError(f"Model name should be either 'modl', 'cdl', or 'tv', but got{cfg_test["model"]["name"]}")

checkpoint = torch.load(run_dir / "model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

metrics_fname = f"test_metrics_{cfg_test["testing"]["noise_variance"]}"
test_model(model, test_loader, device, run_dir, metrics_fname)
