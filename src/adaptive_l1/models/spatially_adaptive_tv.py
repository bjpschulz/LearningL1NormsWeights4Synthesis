import torch
import mrpro
from einops import rearrange


class TVParameterMapNetwork2D(torch.nn.Module):
    r"""A for estimating regularization parameter maps for 2D TV-reconstruction for MRI.

    This is a wrapper for a network block that maps an image of shape (batch, 2, ny, nx) to one
    spatially adaptive regularization parameter map of shape (batch, 1, ny, nx).
    In the forward, the input image, which is expected to follow the MRpro shapes convention, i.e.
    (batch, coils, z, y, x) with coils, z=1 here (for 2D problems), is first reshaped to (batch * coils, 2, ny, nx),
    then fed into the network block.
    The output of the network is then reshaped to be used as weight in the definition of the weighted
    TV functional in the PDHG algorithm.
    """

    def __init__(self, cnn_block: torch.nn.Module) -> None:
        r"""Initialize Parameter Map Network.

        Parameters
        ----------
        cnn_block
            A neural network for estimating the regularization parameter maps.
        """
        super().__init__()
        self.cnn_block = cnn_block

        # raw parameter for global scaling; softplus is used to "activate" it and make it strictly positive
        self._global_scaling = torch.nn.Parameter(
            torch.tensor([-1.0], requires_grad=True)
        )

    @property
    def global_scaling(self):
        return torch.nn.functional.softplus(self._global_scaling, beta=1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""Apply the network to estimate regularization parameter maps.

        Parameters
        ----------
        image
            the image from which the regularization parameter maps should be estimated.
        """
        image = rearrange(torch.view_as_real(image), "batch 1 1 y x ch -> batch ch y x")
        regularization_parameter_map = torch.nn.functional.softplus(
            (self.cnn_block(image).swapaxes(0, 1).unsqueeze(-3).unsqueeze(-3))
        )
        regularization_parameter_map = (
            self.global_scaling * regularization_parameter_map
        )

        return regularization_parameter_map


class SpatiallyAdaptiveTVNet2D(torch.nn.Module):
    r"""Unrolled primal dual hybrid gradient with spatially adaptive regularization parameter maps for TV
    for 2D imaging.

    The network is based on the work
        ..  [KOF2023] Kofler A, Alterkrueger F, Antarou Ba F, Kolbitsch C, Papoutsellis E,
            Schote D, Sirotenko D, Zimmermann F F, Papafitsoros K (2023).
            Learning regularization parameter-maps for variational image
            reconstruction using deep neural networks and algorithm unrolling.
            SIAM Journal on Imaging Sciences, Volume 16, Number 4, p. 2202--2246.

    The forward of the network approximately solves the minimization problem

        :math:`\min_x \frac{1}{2}\| Ax - y\|_2^2 + \| \Lambda_{\theta} \nabla x\|_1`,
    where :math:`A` is the forward linear operator, :math:`\nabla` is the gradient operator,
    and :math:`\Lambda_{\theta}` is a strictly positive regularization parameter map that is estimated from
    an input image with a network :math:`u_{\theta}` with trainable parameters :math:`\theta`.

    N.B. The entire network sticks to the convention of MRpro, i.e. we work with images and k-space data
    of shape (other*, coils, z, y, x).
    """

    def __init__(
        self,
        parameter_map_network: torch.nn.Module,
        n_iterations: int = 128,
    ):
        r"""Initialize Adaptive TV Network.

        Parameters
        ----------
        parameter_map_network
            a network that predicts a regularization parameter map from the input image.
        n_iterations
            number of iterations for the unrolled primal dual hybrid gradient (PDHG) algorithm.

        """
        super().__init__()

        self.parameter_map_network = parameter_map_network
        self.n_iterations = n_iterations

        self.fourier_operator = mrpro.operators.FastFourierOp(dim=(-2, -1))
        self.gradient_operator = mrpro.operators.FiniteDifferenceOp(
            dim=(-2, -1), mode="forward", pad_mode="circular"
        )
        self.g = mrpro.operators.functionals.ZeroFunctional()

        # operator norm of the stacked operator K=[A, \nabla]^T
        stacked_operator_norm = 3.0  # analytically calculated
        self.primal_stepsize = self.dual_stepsize = 0.97 * (1.0 / stacked_operator_norm)

    def forward(
        self,
        initial_image: torch.Tensor,
        kdata: torch.Tensor,
        mask_operator: mrpro.operators.LinearOperator,
        regularization_parameter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct image using an unrolled PDHG algorithm.

        Parameters
        ----------
        initial_image
            initial guess of the solution of the TV problem.
        kdata
            k-space data tensor of the considered problem.
        mask_operator
            mask operator that masks the k-space data.
        regularization_parameter
            regularization parameter to be used in the TV-functional. If set to None,
            it is estimated by the lambda_map_network.
            (can also be a single scalar)

        Returns
        -------
            Image reconstructed by the PDGH algorithm to solve the weighted TV problem.
        """
        # if no regularization parameter map is provided, compute it with the network
        if regularization_parameter is None:
            regularization_parameter = self.parameter_map_network(initial_image)

        l2_norm_squared = 0.5 * mrpro.operators.functionals.L2NormSquared(target=kdata)
        l1_norm = mrpro.operators.functionals.L1NormViewAsReal(
            weight=regularization_parameter
        )
        f = mrpro.operators.ProximableFunctionalSeparableSum(l2_norm_squared, l1_norm)

        forward_operator = mask_operator @ self.fourier_operator

        stacked_operator = mrpro.operators.LinearOperatorMatrix(
            ((forward_operator,), (self.gradient_operator,))
        )

        (solution,) = mrpro.algorithms.optimizers.pdhg(
            f=f,
            g=self.g,
            operator=stacked_operator,
            initial_values=(initial_image,),
            max_iterations=self.n_iterations,
            primal_stepsize=self.primal_stepsize,
            dual_stepsize=self.dual_stepsize,
            tolerance=0.0,
        )
        return solution
