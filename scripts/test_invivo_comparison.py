import mrpro
import torch
from adaptive_l1.data.utils import load_config
from adaptive_l1.data.data_classes import normalize_kspace_data_and_images

import os
from pathlib import Path
import zenodo_get

import matplotlib.pyplot as plt

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

dataset = "15348115"
output_dir = Path(os.getcwd()).parent / "zenodo"

zenodo_get.download(
    record=dataset,
    retry_attempts=8,
    output_dir=output_dir,
    file_glob=("t2_data_2d.mrd",),
    access_token=os.environ.get("ZENODO_TOKEN"),
)  # r: retries


kdata_scanner = mrpro.data.KData.from_file(
    output_dir / "t2_data_2d.mrd",
    mrpro.data.traj_calculators.KTrajectoryCartesian(),
)

super_resolution_factor = 2
ny_low_resolution, nx_low_resolution = (
    kdata_scanner.data.shape[-2],
    kdata_scanner.data.shape[-1],
)
nz, ny_target, nx_target = (
    1,
    int(super_resolution_factor * ny_low_resolution),
    int(super_resolution_factor * nx_low_resolution),
)  # target resolution
n_k1, n_k0 = ny_low_resolution, nx_low_resolution

pad_op = mrpro.operators.ZeroPadOp(
    dim=(-2, -1),
    original_shape=(ny_low_resolution, nx_low_resolution),
    padded_shape=(ny_target, nx_target),
)
(kdata_padded,) = pad_op(kdata_scanner.data)

mask = torch.zeros_like(kdata_padded.abs())
mask[kdata_padded.abs() != 0] = 1

mask_operator = mrpro.operators.CartesianMaskingOp(mask)
fourier_operator = mrpro.operators.FastFourierOp(dim=(-2, -1))

forward_operator = mask_operator @ fourier_operator

(adjoint_reconstruction,) = forward_operator.H(kdata_padded)

kdata_padded, adjoint_reconstruction, _ = normalize_kspace_data_and_images(
    kdata_padded, adjoint_reconstruction, target=None
)

(low_resolution_adjoint_recon,) = fourier_operator.H(pad_op.H(kdata_padded)[0])

with torch.no_grad():
    cdl_recon = model_cdl(
        adjoint_reconstruction.to(device),
        kdata_padded.to(device),
        mask_operator.to(device),
    )

    modl_recon = model_modl(
        adjoint_reconstruction.to(device),
        kdata_padded.to(device),
        mask_operator.to(device),
    )

    tv_recon = model_tv(
        adjoint_reconstruction.to(device),
        kdata_padded.to(device),
        mask_operator.to(device),
    )


fig, ax = plt.subplots(1, 4, figsize=(32, 8))
clim = clim = [0, 0.6 * cdl_recon.abs().max()]
recons_list = [1 / 2 * low_resolution_adjoint_recon, tv_recon, cdl_recon, modl_recon]

titles_list = [
    "Low Resolution IFFT",
    r"TV-$\boldsymbol{\Lambda}$",
    r"CDL-$\boldsymbol{\Lambda}$",
    "MoDL",
]
fontsize = 24
for k, (recon, title) in enumerate(zip(recons_list, titles_list, strict=True)):
    ax[k].imshow(
        recon.abs().squeeze().detach().cpu().rot90(k=-1),
        clim=clim,
        cmap="gray",
    )
    ax[k].set_title(title, fontsize=fontsize)
plt.setp(ax, xticks=[], yticks=[])
fig.savefig("images/invivo_comparison.png", bbox_inches="tight", dpi=300)
