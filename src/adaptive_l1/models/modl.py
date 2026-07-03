import torch
import mrpro
from einops import rearrange


class MoDLBlock(torch.nn.Module):
    """Am implementation of the simple CNN-block (Conv+BN+ReLU, last layer no ReLU)
    as described in the MoDL paper.

    The forward returns residual output: net(x) + x, when shapes are compatible.
    """

    def __init__(
        self,
        n_layers: int = 5,
        n_ch_in: int = 2,
        n_ch_out: int = 2,
        n_filters: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers = []

        if n_layers == 1:
            layers.append(
                torch.nn.Conv2d(
                    n_ch_in, n_ch_out, kernel_size, padding=padding, bias=bias
                )
            )
            layers.append(torch.nn.BatchNorm2d(n_ch_out))
        else:
            layers.append(
                torch.nn.Conv2d(
                    n_ch_in, n_filters, kernel_size, padding=padding, bias=bias
                )
            )
            layers.append(torch.nn.BatchNorm2d(n_filters))
            layers.append(torch.nn.ReLU(inplace=True))

            for _ in range(n_layers - 2):
                layers.append(
                    torch.nn.Conv2d(
                        n_filters,
                        n_filters,
                        kernel_size,
                        padding=padding,
                        bias=bias,
                    )
                )
                layers.append(torch.nn.BatchNorm2d(n_filters))
                layers.append(torch.nn.ReLU(inplace=True))

            layers.append(
                torch.nn.Conv2d(
                    n_filters, n_ch_out, kernel_size, padding=padding, bias=bias
                )
            )
            layers.append(torch.nn.BatchNorm2d(n_ch_out))

        self.net = torch.nn.Sequential(*layers)
        self.in_channels = n_ch_in
        self.out_channels = n_ch_out

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        if image.shape[-3] != 1 or image.shape[-4] != 1:
            raise ValueError(
                f"Expected a single-coil 2D input image (i.e. z=1),"
                f"but got coils={image.shape[-4]},z={image.shape[-3]}."
                f"Please ensure the input image has shape (batch, coils, z, y, x) with coils=z=1 for 2D problems."
            )

        image_in = image.clone()
        image = rearrange(torch.view_as_real(image), "batch 1 1 y x ch -> batch ch y x")

        image_regularized = self.net(image)
        image_regularized = torch.view_as_complex(
            rearrange(
                image_regularized,
                "batch ch y x -> batch 1 1 y x ch",
            ).contiguous()
        )

        if image_regularized.shape != image_in.shape:
            raise RuntimeError(
                f"Residual add requires net(x) and x to have the same shape, "
                f"but got net(x): {tuple(image_regularized.shape)} vs x: {tuple(image_in.shape)}. "
                f"Set out_channels=in_channels (and keep spatial sizes unchanged) to use image_regularized + image_in."
            )

        return image_regularized + image_in


class MoDL(torch.nn.Module):
    """Unrolled Model-based Depp Learning (MoDL) Method for 2D MRI reconstruction based on

        ..  [AGG2018] Aggarwal, H K, Mani M P, Jacob, M (2018)
            MoDL: Model-Based Deep Learning Architecture for Inverse Problems.
            IEEE Transactions on Medical Imaging, Volume 38, Number 2, p. 394--405.

    The following netwrk performs the following operations for a fixed number of iterations
        :math:`z_k = \mathrm{NET}_{\Theta}(x_k)`,
        :math:`x_{k+1} =\argmin_x \frac{1}{2}\| Ax - y\|_2^2 + \frac{\lambda}{2}\| x - z_k\|_2^2`,

    where :math:`A` is the forward linear operator, :math:`\lambda` is the regularization parameter.

    N.B. The entire network sticks to the convention of MRpro, i.e. we work with images and k-space data
    of shape (other*, coils, z, y, x). However, because here showcase the method for 2D problems,
    some processing steps are necessary within the forward method. In particular, we restrict this example,
    to be used with z=1.
    """

    def __init__(
        self,
        cnn_block,
        n_iterations=8,
        normalize_input=False,
    ):
        super().__init__()
        self.cnn_block = cnn_block
        self.n_iterations = n_iterations
        self.normalize_input = normalize_input
        self._regularization_parameter = torch.nn.Parameter(
            torch.tensor(
                -3.0,
            ),
            requires_grad=True,
        )

        self.fast_fourier_operator = mrpro.operators.FastFourierOp(dim=(-2, -1))

    @property
    def regularization_parameter(self):
        return torch.nn.functional.softplus(self._regularization_parameter, beta=1.0)

    def data_consistency(
        self,
        regularized_image: torch.Tensor,
        kdata: torch.Tensor,
        mask_operator: mrpro.operators.LinearOperator,
        regularization_parameter: torch.Tensor,
    ):
        """Apply data-consistency step.

        Args:
            regularized_image (torch.Tensor): image used for Tikhonov regularization
            kdata (torch.Tensor): measured k-space data
            mask_operator (mrpro.operators.LinearOperator): mask operator that masks the k-space data
            regularization_parameter (torch.Tensor): scalara regularization parameter
                for the Tikhonov regularization

        Returns:
            torch.Tensor: Tikhonov-regularized image
        """
        (regularized_kdata,) = self.fast_fourier_operator(regularized_image)

        mask_complement_operator = mrpro.operators.CartesianMaskingOp(
            mask=1.0 - mask_operator.mask
        )

        estimated_kdata = (
            regularization_parameter
            / (1.0 + regularization_parameter)
            * mask_operator(regularized_kdata)[0]
            + mask_complement_operator(regularized_kdata)[0]
            + 1 / (1.0 + regularization_parameter) * kdata
        )

        (solution,) = self.fast_fourier_operator.H(estimated_kdata)

        return solution

    def forward(
        self,
        image: torch.Tensor,
        kdata: torch.Tensor,
        mask_operator: mrpro.operators.LinearOperator,
    ) -> torch.Tensor:
        """Reconstruct image using the MoDL unrolled iterations.

        Parameters
        ----------
        image
            input image tensor.
        kdata
            k-space data tensor of the considered problem.
        mask_operator
            mask operator that masks the k-space data.

        Returns
        -------
            Image reconstructed by the MoDL unrolled iterations.
        """

        for _ in range(self.n_iterations):

            reguarlized_image = self.cnn_block(image)

            image = self.data_consistency(
                reguarlized_image,
                kdata,
                mask_operator,
                self.regularization_parameter,
            )
        return image
