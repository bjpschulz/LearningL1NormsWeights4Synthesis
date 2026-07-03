# %%
import mrpro
import torch
from adaptive_l1.data.utils import load_config
from adaptive_l1.data.utils import read_split_file

from adaptive_l1.data.data_classes import LowFieldMRDataset

import matplotlib.pyplot as plt

from adaptive_l1.testing.statistics import brain_mask

from adaptive_l1.models.utils import (
    define_cdl_model,
    define_tv_model,
    define_modl_model,
)
from adaptive_l1.models.utils import (
    create_cdl_run_directory,
    create_tv_run_directory,
    create_modl_run_directory,
)

import itertools

# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_cdl", type=str, required=True)
parser.add_argument("--config_tv", type=str, required=True)
parser.add_argument("--config_modl", type=str, required=True)
parser.add_argument("--config_data", type=str, required=True)

args = parser.parse_args()

cfg_cdl = load_config(args.config_cdl)
cfg_tv = load_config(args.config_tv)
cfg_modl = load_config(args.config_modl)
cfg_data = load_config(args.config_data)

data_dir = cfg_data["data_dir"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cdl = define_cdl_model(cfg_cdl)
run_dir_cdl = create_cdl_run_directory(cfg_cdl)
checkpoint = torch.load(run_dir_cdl / "model.pt", map_location=device)
model_cdl.load_state_dict(checkpoint["model_state_dict"])

model_tv = define_tv_model(cfg_tv)
run_dir_tv = create_tv_run_directory(cfg_tv)
checkpoint = torch.load(run_dir_tv / "model.pt", map_location=device)
model_tv.load_state_dict(checkpoint["model_state_dict"])

model_modl = define_modl_model(cfg_modl)
run_dir_modl = create_modl_run_directory(cfg_modl)
checkpoint = torch.load(run_dir_modl / "model.pt", map_location=device)
model_modl.load_state_dict(checkpoint["model_state_dict"])

data_dir = cfg_data["data_dir"]
split_dir = cfg_data["split_dir"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_variance = 0.3
n_k1 = 160
n_test = 9

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

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

sample_id = 17
batch = next(itertools.islice(test_loader, sample_id, sample_id + 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kdata = batch["kdata"].to(device)
adjoint = batch["adjoint"].to(device)
mask = batch["mask"].to(device)
target = batch["target"].to(device)
mask_operator = mrpro.operators.CartesianMaskingOp(mask).to(device)

with torch.no_grad():
    cdl_recon = model_cdl(
        adjoint.to(device),
        kdata.to(device),
        mask_operator.to(device),
    )

    modl_recon = model_modl(
        adjoint.to(device),
        kdata.to(device),
        mask_operator.to(device),
    )

    tv_recon = model_tv(
        adjoint.to(device),
        kdata.to(device),
        mask_operator.to(device),
    )

fig, ax = plt.subplots(2, 5, figsize=(40, 16))
clim = clim = [0, 0.6 * target.abs().max()]
cutoff_y, cutoff_x = 5, 30
recons_list = [adjoint, tv_recon, cdl_recon, modl_recon, target]
recons_list = [
    recon[..., cutoff_y:-cutoff_y, cutoff_x:-cutoff_x] for recon in recons_list
]
titles_list = [
    "Adjoint",
    r"TV-$\boldsymbol{\Lambda}$",
    r"CDL-$\boldsymbol{\Lambda}$",
    "MoDL",
    "Target",
]
image_mask = brain_mask(recons_list[-1].abs().squeeze(), 0.1).to(device)
fontsize = 28
for k, (recon, title) in enumerate(zip(recons_list, titles_list, strict=True)):
    ax[0, k].imshow(
        (image_mask * recon).abs().squeeze().detach().cpu(),
        clim=clim,
        cmap="gray",
    )
    ax[0, k].set_title(title, fontsize=fontsize)
    error = image_mask * (recon - recons_list[-1])
    ax[1, k].imshow(
        error.abs().squeeze().detach().cpu(),
        clim=clim,
        cmap="viridis",
    )
    mse = torch.nn.functional.mse_loss(
        torch.view_as_real(image_mask * recon),
        torch.view_as_real(image_mask * recons_list[-1]),
    ).item()
    if k < 4:
        ax[1, k].text(
            0.5,
            0.12,
            f"MSE: {mse:.2e}",
            color="yellow",
            fontsize=24,
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax[1, k].transAxes,
            bbox={
                "facecolor": "black",
                "alpha": 0.8,
                "pad": 1,
            },
        )

plt.subplots_adjust(hspace=0.04, wspace=-0.4)
plt.setp(ax, xticks=[], yticks=[])
fig.savefig("images/models_comparison.png", bbox_inches="tight", dpi=300)
