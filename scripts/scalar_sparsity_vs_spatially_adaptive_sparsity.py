import mrpro
import torch
from adaptive_l1.data.utils import read_split_file
from adaptive_l1.data.data_classes import LowFieldMRDataset
from adaptive_l1.data.utils import load_config

from adaptive_l1.testing.statistics import brain_mask

from adaptive_l1.models.utils import define_cdl_model, create_cdl_run_directory

import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_test", type=str, required=True)
parser.add_argument("--config_data", type=str, required=True)

args = parser.parse_args()

cfg_test = load_config(args.config_test)
cfg_data = load_config(args.config_data)

noise_variance = cfg_test["testing"]["noise_variance"]
if not (isinstance(noise_variance, float) or isinstance(noise_variance, int)):
    raise ValueError(
        f"noise standard deviation should be float or integer, got {noise_variance}"
    )

n_k1 = cfg_test["testing"]["n_k1"]
if not (isinstance(n_k1, float) or isinstance(n_k1, int)):
    raise ValueError(
        f"A single number of n_k1 samples (`float` or `int`) should be "
        f"given for model testing; got {type(n_k1)}"
    )

data_dir = cfg_data["data_dir"]
split_dir = cfg_data["split_dir"]

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

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cdl = define_cdl_model(cfg_test)
run_dir_cdl = create_cdl_run_directory(cfg_test)
checkpoint = torch.load(run_dir_cdl / "model.pt", map_location=device)
model_cdl.load_state_dict(checkpoint["model_state_dict"])

# select sample id 7
sample_id = 7
batch = next(itertools.islice(test_loader, sample_id, sample_id + 1))

kdata = batch["kdata"].to(device)
adjoint = batch["adjoint"].to(device)
mask = batch["mask"].to(device)
target = batch["target"].to(device)

mask_operator = mrpro.operators.CartesianMaskingOp(mask=mask)

with torch.no_grad():
    # reconstruction with spatially adaptive sparsity level map
    spatially_adaptive_recon = model_cdl(
        adjoint,
        kdata,
        mask_operator,
    )


def grid_search(
    sparsity_parameters: torch.Tensor,
    low_pass_parameters_raw: torch.Tensor,
    model: torch.nn.Module,
    initial_image: torch.Tensor,
    kdata: torch.Tensor,
    mask_operator: mrpro.operators.LinearOperator,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform a line search to pick the best scalar regularization parameter."""
    mse_matrix = torch.zeros(len(low_pass_parameters_raw), len(sparsity_parameters))

    for i, low_pass_parameter_raw in enumerate(low_pass_parameters_raw):
        model._low_pass_filtering_parameter = torch.nn.Parameter(
            low_pass_parameter_raw.to(target.device)
        )
        reconstructions_list = [
            model(initial_image, kdata, mask_operator, sparsity_parameter)
            for sparsity_parameter in sparsity_parameters
        ]
        mse_values = torch.tensor(
            [
                torch.nn.functional.mse_loss(
                    torch.view_as_real(recon), torch.view_as_real(target)
                )
                for recon in reconstructions_list
            ]
        )
        mse_matrix[i, :] = torch.tensor(mse_values)

    index = mse_matrix.argmin()
    row_col = torch.unravel_index(index, mse_matrix.shape)
    row = row_col[0].item()
    col = row_col[1].item()

    model._low_pass_filtering_parameter = torch.nn.Parameter(
        low_pass_parameters_raw[row].to(target.device)
    )
    best_sparsity_parameter, best_low_pass_parameter = (
        sparsity_parameters[col],
        model.low_pass_filtering_parameter,
    )
    best_recon = model(initial_image, kdata, mask_operator, best_sparsity_parameter)

    return mse_matrix, best_recon, best_sparsity_parameter, best_low_pass_parameter


sparsity_parameters = torch.linspace(0.05, 0.2, 8)
low_pass_parameters_raw = torch.linspace(3.0, 5.0, 8)

with torch.no_grad():
    # grid search to obtain the best reconstruction for scalar
    # regularization parameter
    mse_matrix, best_scalar_recon, best_sparsity_parameter, best_low_pass_parameter = (
        grid_search(
            sparsity_parameters.to(device),
            low_pass_parameters_raw.to(device),
            model_cdl,
            adjoint,
            kdata,
            mask_operator,
            target,
        )
    )

    # reconstruction with spatially adaptive sparsity level map
    spatially_adaptive_recon = model_cdl(
        adjoint,
        kdata,
        mask_operator,
    )

fig, ax = plt.subplots(figsize=(8, 6))
scale = 1e4
display_matrix = mse_matrix * scale
im = ax.imshow(display_matrix, origin="lower", aspect="auto", cmap="viridis")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"MSE ($\times 10^{-4}$)")

ax.set_xticks(torch.arange(len(sparsity_parameters)))
ax.set_yticks(torch.arange(len(low_pass_parameters_raw)))

ax.set_xticklabels(
    [f"{sparsity_param:.2f}" for sparsity_param in sparsity_parameters.tolist()]
)
low_pass_parameters = [
    torch.nn.functional.softplus(low_pass_param, beta=1.0).item()
    for low_pass_param in low_pass_parameters_raw
]
ax.set_yticklabels([f"{low_pass_param:.2f}" for low_pass_param in low_pass_parameters])

ax.set_xlabel(r"Scalar Sparsity Parameter Value $\lambda$", fontsize=12)
ax.set_ylabel(r"Scalar Low Pass Parameter Value $\beta$", fontsize=12)


for i in range(len(low_pass_parameters)):
    for j in range(len(sparsity_parameters)):
        ax.text(
            j,
            i,
            f"{display_matrix[i, j]:.4f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )

min_flat_idx = torch.argmin(mse_matrix)
min_row, min_col = divmod(min_flat_idx.item(), mse_matrix.shape[1])

rect = Rectangle(
    (min_col - 0.5, min_row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3
)

ax.add_patch(rect)

plt.tight_layout()
plt.show()
fig.savefig(
    "images/grid_search_scalar_reg_parameters.png", bbox_inches="tight", dpi=300
)


fig, ax = plt.subplots(1, 4, figsize=(1 * 15, 4 * 15))
recons_list = [
    adjoint,
    best_scalar_recon,
    spatially_adaptive_recon,
    target,
]
image_mask = brain_mask(recons_list[-1].abs().squeeze(), 0.1).to(device)
recons_list = [image_mask * recon for recon in recons_list]
cutoff_y, cutoff_x = 50, 40
best_sparsity_parameter_str = (
    r"$\lambda_{\mathrm{best}}=$" + f"{best_sparsity_parameter:.3f}"
)
best_low_pass_parameter_str = (
    r"$\beta_{\mathrm{best}}=$" + f"{best_low_pass_parameter:.3f}"
)
cdl_scalar_title = (
    r"$\mathrm{CDL}{-}\lambda_{\mathrm{best}},$"
    + "\n"
    + best_sparsity_parameter_str
    + "\n"
    + best_low_pass_parameter_str
)
titles_list = ["adjoint", cdl_scalar_title, r"CDL-$\boldsymbol{\Lambda}$", "target"]
for k, (recon_, title) in enumerate(zip(recons_list, titles_list)):
    ax[k].imshow(
        recon_.abs()[0].squeeze().detach().cpu()[cutoff_y:, cutoff_x:-cutoff_x],
        cmap="gray",
        clim=[0, 0.4 * target.abs().max()],
    )
    mse = torch.nn.functional.mse_loss(
        torch.view_as_real(recon_), torch.view_as_real(recons_list[-1])
    ).item()
    ax[k].set_title(title)
    if k < 3:
        ax[k].text(
            0.5,
            0.12,
            f"MSE: {mse:.2e}",
            color="yellow",
            fontsize=24,
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax[k].transAxes,
            bbox={
                "facecolor": "black",
                "alpha": 0.8,
                "pad": 1,
            },
        )

plt.setp(ax, xticks=[], yticks=[])
fig.savefig(
    "images/scalar_vs_spatially_adaptive_sparsity_level_map.png",
    bbox_inches="tight",
    dpi=300,
)
