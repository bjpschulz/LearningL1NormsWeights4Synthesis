import mrpro
import torch
from adaptive_l1.data.utils import read_split_file
from adaptive_l1.data.data_classes import LowFieldMRDataset

from adaptive_l1.models.utils import define_cdl_model, create_cdl_run_directory

import itertools


from adaptive_l1.data.utils import load_config

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


sample_id = 7
batch = next(itertools.islice(test_loader, sample_id, sample_id + 1))


kdata = batch["kdata"].to(device)
adjoint = batch["adjoint"].to(device)
mask = batch["mask"].to(device)
target = batch["target"].to(device)

mask_operator = mrpro.operators.CartesianMaskingOp(mask=mask)


def line_search(
    sparsity_parameters: torch.Tensor,
    model: torch.nn.Module,
    initial_image: torch.Tensor,
    kdata: torch.Tensor,
    mask_operator: mrpro.operators.LinearOperator,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform a line search to pick the best scalar regularization parameter for TV."""
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
    best_sparsity_parameter = sparsity_parameters[torch.argmin(mse_values)]
    best_recon = reconstructions_list[torch.argmin(mse_values)]
    return mse_values, best_recon, best_sparsity_parameter


n_figures = 12
figsize_xy = 5
fontsize = 36

n_filters = model_cdl.kernel.shape[0]

sparsity_parameters = torch.linspace(0.08, 0.2, 8)
with torch.no_grad():

    sparsity_level_map = model_cdl.parameter_map_network(adjoint)

    _, _, best_scalar_sparsity_parameter = line_search(
        sparsity_parameters, model_cdl, adjoint, kdata, mask_operator, target
    )

    image_low_passed = model_cdl.low_pass_filtering_image(
        adjoint, model_cdl.low_pass_filtering_parameter
    )

    kdata_high_passed = model_cdl.high_pass_filtering_kdata(
        kdata, mask_operator @ model_cdl.fourier_operator, image_low_passed
    )

    initial_sparse_code = torch.zeros(
        n_filters,
        *adjoint.shape,
        device=adjoint.device,
        dtype=adjoint.dtype,
    ).to(device)

    # reconstruction with spatially adaptive sparsity level map
    sparse_code_adaptive_sparsity = model_cdl.solve_sparse_coding_problem(
        initial_sparse_code,
        kdata_high_passed,
        mask_operator,
        sparsity_level_map,
        max_iterations=model_cdl.n_iterations,
    )
    sparse_code_scalar_sparsity = model_cdl.solve_sparse_coding_problem(
        initial_sparse_code,
        kdata_high_passed,
        mask_operator,
        torch.tensor(0.157),
        max_iterations=model_cdl.n_iterations,
    )


l1_norms_sparse_codes_adaptive = (
    sparse_code_adaptive_sparsity.abs().squeeze().sum(dim=(-2, -1))
)
l1_norms_sparse_codes_scalar = (
    sparse_code_scalar_sparsity.abs().squeeze().sum(dim=(-2, -1))
)

sorted_l1_norms_sparse_codes_adaptive, sorted_l1_norms_indices = torch.sort(
    l1_norms_sparse_codes_adaptive, descending=True
)
sorted_l1_norms_sparse_codes_scalar, _ = torch.sort(
    l1_norms_sparse_codes_scalar, descending=True
)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    sorted_l1_norms_sparse_codes_adaptive.tolist(),
    linewidth=4,
    label=r"Spatial $\boldsymbol{\Lambda}$",
)
ax.plot(
    sorted_l1_norms_sparse_codes_scalar.tolist(), linewidth=4, label=r"Scalar $\lambda$"
)
ax.set_xlabel(r"Sorted Indices $k$")
positions = torch.arange(len(sorted_l1_norms_indices))
ax.set_ylabel(r"$\ell_1$-Norm of $k$-th Sparse Code")
ax.yaxis.grid(color="gray", linestyle="dashed", alpha=0.3)
ax.legend()
fig.savefig("images/l1_norms.png", bbox_inches="tight", dpi=300)


indices = (
    sorted_l1_norms_indices.tolist()[: int(n_figures / 2)]
    + sorted_l1_norms_indices.tolist()[-int(n_figures / 2) :]
)


fig, ax = plt.subplots(
    3, len(indices), figsize=(len(indices) * figsize_xy, 4 * figsize_xy)
)

row_titles = [
    "Sparse Codes \n(Real Part)",
    "Sparse Codes  \n(Imag Part)",
    "Sparsity Level Maps \n + Conv. Filters",
]

factor_sparse_codes = 1e-4
sparse_code_mean = sparse_code_adaptive_sparsity.abs().mean().item()


max_value = max(
    sparse_code_adaptive_sparsity.real.max().item(),
    sparse_code_adaptive_sparsity.imag.max().item(),
)
min_value = min(
    sparse_code_adaptive_sparsity.real.min().item(),
    sparse_code_adaptive_sparsity.imag.min().item(),
)
clim_sparse_codes = [factor_sparse_codes * min_value, factor_sparse_codes * max_value]
clim_sparsity_level_maps = [0.0, 0.4 * sparsity_level_map.max().item()]
lambda_im = None

cutoff_y, cutoff_x = 50, 50
for k, idx in enumerate(indices):
    ax[0, k].imshow(
        sparse_code_adaptive_sparsity.real.squeeze()[idx][
            cutoff_y:-cutoff_y, cutoff_x:-cutoff_x
        ]
        .detach()
        .cpu(),
        clim=clim_sparse_codes,
        cmap="gray",
    )

    ax[1, k].imshow(
        sparse_code_adaptive_sparsity.imag.squeeze()[idx][
            cutoff_y:-cutoff_y, cutoff_x:-cutoff_x
        ]
        .detach()
        .cpu(),
        clim=clim_sparse_codes,
        cmap="gray",
    )

    lambda_im = ax[2, k].imshow(
        sparsity_level_map.squeeze()[idx][cutoff_y:-cutoff_y, cutoff_x:-cutoff_x]
        .detach()
        .cpu(),
        clim=clim_sparsity_level_maps,
        cmap="inferno",
    )

    axins = inset_axes(
        ax[2, k], width="25%", height="25%", loc="upper left", borderpad=1
    )
    axins.imshow(
        model_cdl.kernel[idx].squeeze().detach().cpu(), clim=[-0.4, 0.4], cmap="viridis"
    )
    axins.axis("off")

for i, title in enumerate(row_titles):
    ax[i, 0].set_ylabel(title, fontsize=fontsize, rotation=90, labelpad=35, va="center")

plt.setp(ax, xticks=[], yticks=[])

plt.subplots_adjust(
    left=0.06, right=0.99, bottom=0.12, top=0.98, wspace=-0.04, hspace=-0.3
)

# horizontal colorbar spanning the full figure width
cbar_ax = fig.add_axes([0.06, 0.14, 0.93, 0.025])
cbar = fig.colorbar(lambda_im, cax=cbar_ax, orientation="horizontal")
cbar.ax.tick_params(labelsize=fontsize * 0.85)
fig.savefig("images/sparsity_level_maps.png", bbox_inches="tight", dpi=300)
