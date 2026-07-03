import mrpro
import torch
from adaptive_l1.data.utils import read_split_file
from adaptive_l1.data.data_classes import LowFieldMRDataset
from adaptive_l1.data.utils import load_config

import itertools
import matplotlib.pyplot as plt

config_data = "/echo/kofler01/projects/public_repositories/LearningL1NormsWeights4Synthesis/configs/data.yaml"
cfg_data = load_config(config_data)

noise_variance = 0.4
n_k1 = 160
n_validation = 9

data_dir = cfg_data["data_dir"]
split_dir = cfg_data["split_dir"]

validation_files = read_split_file(
    data_dir=data_dir,
    split_file=split_dir + "fastmri_validation.txt",
)[:n_validation]

validation_image_data = mrpro.phantoms.FastMRIImageDataset(
    path=validation_files,
    coil_combine=True,
)

base_seed = 42

validation_data = LowFieldMRDataset(
    image_dataset=validation_image_data,
    noise_variance=noise_variance,
    n_k1=n_k1,
    base_seed=base_seed,
)

validation_loader = torch.utils.data.DataLoader(
    validation_data, batch_size=1, shuffle=False
)

sample_id = 6
batch = next(itertools.islice(validation_loader, sample_id, sample_id + 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kdata = batch["kdata"].to(device)
adjoint = batch["adjoint"].to(device)
mask = batch["mask"].to(device)
target = batch["target"].to(device)

fig, ax = plt.subplots(1, 3, figsize=(24, 8))
tensors_list = [kdata, adjoint, target]
titles_list = ["Masked K-Space Data", "Adjoint Recon", "Target"]
fontsize = 20
for k, (data, title) in enumerate(zip(tensors_list, titles_list)):
    factor = 0.05 if k == 0 else 0.4
    clim = [0, factor * data.abs().max()]
    ax[k].imshow(data.abs().squeeze().detach().cpu(), clim=clim, cmap="gray")
    ax[k].set_title(title, fontsize=fontsize)
plt.setp(ax, xticks=[], yticks=[])
