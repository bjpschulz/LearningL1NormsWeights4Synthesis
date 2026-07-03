import torch
import mrpro


class LowFieldMRDataset(torch.utils.data.Dataset):
    """LowFieldMRDataset based on the FastMRI Image Dataset.

    This dataset returns a dictionary containing k-space data, adjoint reconstruction, 
    a binary mask and a target image to be used for supervised training.

    The target images are complex-valued and have shape ``(1, 1, 1, 320, 320)``. The binary
    masks, the adjoint and the k-space data have the same shape.

    The fastMRI data has to be downloaded beforehand. See https://fastmri.med.nyu.edu/ for more information.
    """

    def __init__(
        self,
        image_dataset: torch.utils.data.Dataset,
        noise_variance: float | tuple | list,
        n_k1: int | list,
        transform = None,
        base_seed=42,
    ):  
        """Initialize the dataset.

        Parameters
        ----------
        image_dataset : torch.utils.data.Dataset
            The fastMRI dataset given by mrpro.phantoms.FastMRIImageDataset
        noise_variance : float, tuple or list
            The variance of the complex-valued noise that is added to the k-space data.
            If `float`, the noise-variance is the number given.
            If `tuple`, the tuple must be (low, high) with low < high and the noise-variance
                is uniformly sampled from the interval (low, high).
            If `list`, the noise variance is sampled from the elements of the list.
        n_k1 : int or list
            The number of readout points sampled in the k-space data, see also mrpro.data.KData.
            if `int`, this is the set parameter.
            If `list`, the value is randomly picked from the elements of the list.
        transform: 
            A composition of transforms to potentially apply to the target image for data
            augmentation purposes.
        """
        
        self.image_dataset = image_dataset
        self.noise_variance = noise_variance
        self.n_k1 = n_k1
        self.base_seed = base_seed
        self.epoch = 0
        self.transform = transform

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        image = self.image_dataset[idx]
        
        if self.transform is not None:
            image = self.transform(image)

        seed = self.base_seed + self.epoch * len(self) + idx
        generator = mrpro.utils.RandomGenerator(seed)

        noise_variance = sample_parameter(self.noise_variance, generator)
        n_k1 = sample_parameter(self.n_k1, generator)

        mask_operator = create_mask_operator(image, n_k1)

        kdata_noisy, adjoint = generate_kspace_data_and_adjoint(
            image, mask_operator, noise_variance, generator
        )
        
        kdata_noisy, adjoint, image = normalize_kspace_data_and_images(kdata_noisy, adjoint, image)

        return {
            "kdata": kdata_noisy.squeeze(0), #remove unnecessary batch dimension
            "adjoint": adjoint.squeeze(0),
            "mask": mask_operator.mask.squeeze(0),
            "target": image.squeeze(0),
        }

def sample_parameter(spec: int | float | tuple | list, generator: torch.Generator):
    """Given a parameter specification, sample a value according to the specification.

    Parameters:
        spec: the parameter specification, which can be:
            - a single value (int or float), which is returned as is
            - a tuple of two values (low, high), which is interpreted as an interval for uniform sampling
            - a list of values, which is interpreted as a set of discrete values to sample from
        generator: a torch random generator for reproducibility

    """
    if isinstance(spec, (int, float)):
        return spec

    if isinstance(spec, dict):
        if spec["low"] >= spec["high"]:
            raise ValueError(
                "Invalid parameter specification: low should be smaller than high", 
                f"got {spec["low"]} and {spec["high"]}"
            )
        low, high = spec["low"], spec["high"]

        if any(type(param) is float for param in [low, high]):
            value = generator._rand(size=(1,), low=low, high=high)
        if all(type(param) is int for param in [low, high]):
            value = generator._randint(size=(1,), low=low, high=high)
        return value

    if isinstance(spec, list):
        # list is interpreted as a set of discrete values to sample from
        index = generator._randint(size=(1,), low=0, high=len(spec) - 1)
        return spec[index]

    raise ValueError(f"Unsupported parameter specification: {spec}")


def create_mask_operator(
    image: torch.Tensor, n_k1: int
) -> mrpro.operators.CartesianMaskingOp:
    """Create a mask operator representing a low-resoulution acquisition.

    Parameters:
        image (torch.Tensor): the image for which the mask should be created, shape (others*, coils, z, y, x)
        n_k1 (int): the number of phase encoding samples
    Returns:
        mrpro.operators.CartesianMaskingOp: the mask operator
    """
    ny, nx = image.shape[-2:]
    if n_k1 > image.shape[-2]:
        raise ValueError(
            f"number of phase encoding points should be smaller or equal to ny, got n_k1={n_k1}, while ny={ny}"
        )
    encoding_matrix = mrpro.data.SpatialDimension(
        z=1, y=ny, x=nx
    )  # only 2D are considered here

    n_k0 = n_k1  # here, we only assume an isotropic scans, i.e. nky=nkx
    traj = mrpro.data.traj_calculators.KTrajectoryCartesian()(
        n_k0=int(n_k0),
        k0_center=int(n_k0 // 2),
        k1_idx=torch.arange(-n_k1 // 2, n_k1 // 2)[..., None, None, :, None],
        k1_center=0,
        k2_idx=torch.tensor(0),
        k2_center=0,
    )
    mask_operator = mrpro.operators.CartesianMaskingOp.from_trajectory(
        traj, encoding_matrix=encoding_matrix
    ).to(image.device)

    return mask_operator


def add_kdata_noise(
    kdata: torch.Tensor,
    mask_operator: mrpro.operators.CartesianMaskingOp,
    noise_variance: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Add Gaussian noise to kspace data.

    Parameters:
        kdata (torch.Tensor): the kspace data to which noise should be added, shape (others*, coils, z, y, x)
        mask_operator (mrpro.operators.CartesianMaskingOp): the mask operator representing the sampling pattern
        noise_variance (float): the variance of the Gaussian noise to be added
        generator (torch.Generator): a torch random generator for reproducibility

    Returns:
        torch.Tensor: the noisy kspace data
    """

    kdata_real = kdata.real
    kdata_imag = kdata.imag

    noise_real = generator.randn_like(kdata_real)
    noise_imag = generator.randn_like(kdata_imag)

    kdata_real = kdata_real + torch.sqrt(noise_variance / 2) * noise_real
    kdata_imag = kdata_imag + torch.sqrt(noise_variance / 2) * noise_imag

    kdata = kdata_real + 1j * kdata_imag

    kdata = kdata * mask_operator.mask

    return kdata

def normalize_kspace_data_and_images(
    kdata: torch.Tensor,
    adjoint: torch.Tensor | None,
    target: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Normalize k-space data and (possibly) adjoint and target images."""
    factor = 1.0 / adjoint.abs().amax(dim=(-2,-1),keepdim=True)
    
    kdata *= factor
    if adjoint is not None:
        adjoint *= factor
    if target is not None:
        target *= factor
    return kdata, adjoint, target

def generate_kspace_data_and_adjoint(
    image: torch.Tensor,
    mask_operator: mrpro.operators.CartesianMaskingOp,
    noise_variance: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate noisy kspace data and adjoint reconstruction for a given image and sampling pattern.

    Args:
        image (torch.Tensor): target image, shape (others*, coils, z, y, x)
        mask_operator (mrpro.operators.CartesianMaskingOp): the mask defining the sampling pattern
        noise_variance (float): noise variance to be added to the kspace data
        generator (torch.Generator): a torch random generator for reproducibility

    Returns:
        tuple[torch.Tensor, torch.Tensor]: noisy kspace data and adjoint reconstruction
    """

    fourier_operator = mrpro.operators.FastFourierOp(dim=(-2, -1)).to(image.device)

    (kdata_full,) = fourier_operator(image)
    
    signal_rms = kdata_full.abs().pow(2).mean(
        dim=(-2, -1), keepdim=True
    ).sqrt()
    
    noise_variance = noise_variance * signal_rms.square()
    
    (kdata,) = mask_operator(kdata_full)

    kdata_noisy = add_kdata_noise(
        kdata, mask_operator, noise_variance, generator
    )
    
    forward_operator = mask_operator @ fourier_operator

    (adjoint,) = forward_operator.H(kdata_noisy)

    return kdata_noisy, adjoint
