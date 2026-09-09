import torch
import mrpro
from einops import rearrange


class ConvSynthesisParameterMapNetwork2D(torch.nn.Module):
    r"""A network for estimating sparsity level parameter maps for 2D Convolutional Synthesis based regularizatzion
    for MRI.
    """

    def __init__(
        self,
        cnn_block: torch.nn.Module,
        upper_bound: float = 0.5,
        sigmoid_beta: float = 8.0,
    ) -> None:
        r"""Initialize Sparsity Level Parameter Map Network.

        Parameters
        ----------
        cnn_block
            A neural network for estimating the sparsity level parameter maps.
        upper_bound
            upper bound to be imposed on the obtainable sparsity level maps.
        """
        super().__init__()
        self.cnn_block = cnn_block

        # upper bound of the sparsity level maps
        self.register_buffer("upper_bound", torch.tensor(upper_bound))

        self.sigmoid_beta = sigmoid_beta  # determines the "slope" of the sigmoid

    def forward(
        self, image: torch.Tensor, conv_dictionary_op: mrpro.operators.LinearOperator | None = None # parameter can be either a LinearOperator or None
    ) -> torch.Tensor:
        r"""Apply the network to estimate sparsity level parameter maps.

        Parameters
        ----------
        image
            the image from which the sparsity level parameter maps should be estimated.
        conv_dictionary_op
            the convolutional dictionary operator. Unused here, part of the interface
            shared with `FilterwiseConvSynthesisParameterMapNetwork2D`.
        """
        image = rearrange(torch.view_as_real(image), "batch 1 1 y x ch -> batch ch y x")
        regularization_parameter_map = self.cnn_block(image)

        regularization_parameter_map = (
            regularization_parameter_map.swapaxes(0, 1).unsqueeze(-3).unsqueeze(-3)
        )
        regularization_parameter_map = self.upper_bound / (
            1.0 + torch.exp(-self.sigmoid_beta * regularization_parameter_map)
        )
        return regularization_parameter_map


class FilterwiseConvSynthesisParameterMapNetwork2D(torch.nn.Module):
    r"""A network for estimating sparsity level parameter maps for 2D Convolutional Synthesis based regularization
    for MRI from the sparse-code-like representation of the input image.

    For each convolutional dictionary filter, the real and imaginary part of the analysis of the input
    image (:math:`D^H x_0`) are stacked to a two-channel image and the same 2-to-1 channel CNN is applied
    to each of them. As the CNN is independent of the number of dictionary filters, the convolutional
    dictionary can be exchanged at training or inference time.
    """

    def __init__(
        self,
        cnn_block: torch.nn.Module,
        upper_bound: float = 0.5,
        sigmoid_beta: float = 8.0,
    ) -> None:
        r"""Initialize Sparsity Level Parameter Map Network.

        Parameters
        ----------
        cnn_block
            A neural network for estimating the sparsity level parameter maps,
            taking a two-channel (real/imaginary part) input image.
        upper_bound
            upper bound to be imposed on the obtainable sparsity level maps.
        sigmoid_beta
            determines the "slope" of the sigmoid.
        """
        super().__init__()
        self.cnn_block = cnn_block

        # upper bound of the sparsity level maps
        self.register_buffer("upper_bound", torch.tensor(upper_bound))

        self.sigmoid_beta = sigmoid_beta

    def forward(
        self, image: torch.Tensor, conv_dictionary_op: mrpro.operators.ConvSynthesisDictionaryOp
    ) -> torch.Tensor:
        r"""Apply the network to estimate sparsity level parameter maps.

        Parameters
        ----------
        image
            the image from which the sparsity level parameter maps should be estimated.
        conv_dictionary_op
            the convolutional dictionary operator defining the dictionary the parameter
            maps are estimated for.
        """
        # apply the analysis of the convolutional dictionary to the image; the
        # analysis operator is used instead of the adjoint of the synthesis
        # operator as it does not rely on autograd and matches the multi-dictionary
        # convention of analyzing with the (unflipped) filters
        analysis_op = mrpro.operators.ConvAnalysisDictionaryOp(
            kernel=conv_dictionary_op.kernel, pad_mode=conv_dictionary_op.pad_mode
        )
        (sparse_code_like,) = analysis_op(image)
        n_filters, n_batch = sparse_code_like.shape[0], image.shape[0]

        # stack the real and imaginary part of the sparse-code-like map of each
        # filter to a two-channel image, moving the filter index to the batch
        sparse_code_two_channel = rearrange(
            torch.view_as_real(sparse_code_like),
            "filter batch 1 1 y x ch -> (filter batch) ch y x",
        )

        # estimate one sparsity level map per filter with the 2-to-1 channel CNN
        regularization_parameter_map = self.cnn_block(sparse_code_two_channel)

        # bring back the filter dimension; the resulting real-valued map matches
        # the complex sparse code layout (filter, batch, 1, 1, y, x), weighting
        # real and imaginary part of the sparse code in the same way
        regularization_parameter_map = rearrange(
            regularization_parameter_map,
            "(filter batch) 1 y x -> filter batch 1 1 y x",
            filter=n_filters,
            batch=n_batch,
        )

        regularization_parameter_map = self.upper_bound / (
            1.0 + torch.exp(-self.sigmoid_beta * regularization_parameter_map)
        )
        return regularization_parameter_map


class SpatiallyAdaptiveConvSynthesisNet2D(torch.nn.Module):
    r"""Unrolled FISTA with spatially adaptive regularization parameter maps for convolutional synthesis
    regularization for 2D imaging.

    The network is based on the work
        ..  [KOF2025] Kofler A, Calatroni L, Kolbitsch C,  Papafitsoros K (2025)
            Learning Spatially Adaptive l1-Norms Weights for Convolutional Synthesis Regularization.
            Proceedings of the 33rd IEEE International Conference on Signal Processing (EUSIPCO).


    Solves the minimization problems
        :math:`x_{low}:=\argmin_x \frac{1}{2}\| x - x0\|_2^2 + \frac{\beta}{2}\| \nabla x\|_2^2`,
        :math:`x^{\ast}:=D s^{\ast}` + x_{low},
        :math:`s^{\ast}:=\argmin_s \frac{1}{2}\| ADs - y^{\prime}\|_2^2 + \| \Lambda_{\theta} s\|_1`,
    where :math:`A` is the forward linear operator, :math:`\nabla` is the gradient operator,
    :math:`D` is the convolutional dictionary operator, :math:`y^{\prime}` is the high-passed k-space data
    and :math:`\Lambda_{\theta}` are strictly positive sparsity level maps that are estimated from an
    input image :math:`x_0:=A^H y` with a network :math:`u_{\theta}` with trainable parameters :math:`\theta`.

    N.B. The entire network sticks to the convention of MRpro, i.e. we work with images and k-space data
    of shape (other*, coils, z, y, x). However, because here showcase the method for 2D problems,
    some processing steps are necessary within the forward method. In particular, we restrict this example,
    to be used with z=1.
    """

    def __init__(
        self,
        kernel: torch.tensor,
        parameter_map_network: torch.nn.Module,
        n_iterations: int = 64,
    ):
        r"""Initialize Adaptive Conv Synthesis Network.

        Parameters
        ----------
        kernel
            convolutionl kernel defining the convolutional synthesis dicitonary operator.
        parameter_map_network
            a network that predicts a regularization parameter map from the input image.
        n_iterations
            number of iterations for the unrolled acclerated proximal gradient (FISTA) algorithm.

        """
        super().__init__()

        self.parameter_map_network = parameter_map_network
        self.n_iterations = n_iterations

        self.fourier_operator = mrpro.operators.FastFourierOp(dim=(-2, -1))
        self.gradient_operator = mrpro.operators.FiniteDifferenceOp(
            dim=(-2, -1), mode="forward", pad_mode="circular"
        )
        self.identity_op = mrpro.operators.IdentityOp()

        self._low_pass_filtering_parameter = torch.nn.Parameter(
            torch.tensor([2.0], requires_grad=True)
        )

        self.register_buffer("_kernel", kernel.clone().detach())
        self.register_buffer("operator_norm", torch.empty(()))

        self._set_conv_dictionary_operator()
        self._compute_operator_norm()

    @property
    def kernel(self) -> torch.Tensor:
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel: torch.Tensor) -> None:
        self.set_kernel(new_kernel)

    def set_kernel(self, new_kernel: torch.Tensor) -> None:
        """Update convolutional synthesis kernel and recompute dependent quantities."""
        new_kernel = new_kernel.detach().to(
            device=self._kernel.device,
            dtype=self._kernel.dtype,
        )

        if new_kernel.shape != self._kernel.shape:
            self._kernel.resize_(new_kernel.shape)

        self._kernel.copy_(new_kernel)

        self._set_conv_dictionary_operator()

        # set the operator norm of the convolutional dictionary.
        # note that here, we only use an upper bound of the actual operator norm, i.e.
        # ||A|| = ||S F D || \leq ||D||, since ||S||=||F|| = 1.
        # This allows to quickly recompute the operator norm for different dictionary filters.
        self._compute_operator_norm()

    def _set_conv_dictionary_operator(self) -> None:
        """Rebuild the convolutional synthesis dictionary operator."""
        self.conv_dictionary_op = mrpro.operators.ConvSynthesisDictionaryOp(
            kernel=self.kernel,
            pad_mode="circular",
        )

    def _compute_operator_norm(self) -> None:
        """Compute and store the convolutional dictionary operator norm."""
        dummy_sparse_code = self._make_dummy_sparse_code_for_operator_norm()
        # the adjoint of the circular padding used in the operator norm estimation
        # is implemented via autograd and therefore requires grad mode to be enabled
        with torch.autograd.set_grad_enabled(True):
            operator_norm = self.conv_dictionary_op.operator_norm(
                initial_value=dummy_sparse_code,
                dim=None,
                max_iterations=16,
            )

        self.operator_norm.resize_as_(operator_norm)
        self.operator_norm.copy_(operator_norm.detach())

    def _make_dummy_sparse_code_for_operator_norm(self) -> torch.Tensor:
        """Create a small dummy sparse code for estimating the dictionary operator norm."""
        n_filters = self.kernel.shape[0]

        return torch.randn(
            n_filters,
            1,
            1,
            int(2 * self.kernel.shape[-2]),
            int(2 * self.kernel.shape[-1]),
            device=self.kernel.device,
            dtype=self.kernel.dtype,
        )

    @property
    def low_pass_filtering_parameter(self):
        """The positive low-pass parameter."""
        return torch.nn.functional.softplus(
            self._low_pass_filtering_parameter, beta=1.0
        )

    def low_pass_filtering_image(
        self, image: torch.Tensor, low_pass_filtering_parameter: torch.Tensor
    ) -> torch.Tensor:
        """Low pass filter the image.

        Args:
            image (torch.Tensor): image to be low-pass filtered
            low_pass_filtering_parameter (torch.Tensor): parameter for low-pass filtering

        Returns:
            torch.Tensor: the low-pass filtered image
        """
        (image_low_passed,) = mrpro.algorithms.optimizers.cg(
            self.identity_op
            + low_pass_filtering_parameter
            * self.gradient_operator.H
            @ self.gradient_operator,
            right_hand_side=image,
            initial_value=image,
            max_iterations=12,
        )
        return image_low_passed

    def high_pass_filtering_kdata(
        self,
        kdata: torch.Tensor,
        forward_operator: mrpro.operators.LinearOperator,
        image_low_passed: torch.Tensor,
    ) -> torch.Tensor:
        """High pass filter the kdata.

        Args:
            kdata (torch.Tensor): kdata to be high-pass filtered
            image_low_passed (torch.Tensor): low-pass filtered image

        Returns:
            torch.Tensor: the high-pass filtered kdata
        """
        return kdata - forward_operator(image_low_passed)[0]

    def solve_sparse_coding_problem(
        self,
        initial_sparse_code: torch.Tensor,
        kdata_high_passed: torch.Tensor,
        mask_operator: mrpro.operators.LinearOperator,
        regularization_parameter: torch.Tensor,
        max_iterations: int = 64,
    ):
        """Reconstruct sparse codes of the problem
            :math:`\min_s \frac{1}{2}\| A D s - y'\|_2^2 + \| \Lambda_{\theta} s\|_1`,

        using an unrolled FISTA algorithm.

        Parameters
        ----------
        initial_sparse_code
            initial guess of the solution of the problem.
        kdata_high_passed
            high-passed k-space data tensor of the considered problem.
        mask_operator
            mask operator that masks the k-space data.
        regularization_parameter
            regularization parameter (map) to be used in the l1-norm functional
        max_iterations
            number of unrolled iterations

        Returns
        -------
            solution of the sparse coding problem.
        """
        forward_operator = (
            mask_operator @ self.fourier_operator @ self.conv_dictionary_op
        )

        l2_norm_squared = 0.5 * mrpro.operators.functionals.L2NormSquared(
            target=kdata_high_passed, divide_by_n=False
        )
        f = l2_norm_squared @ forward_operator
        g = mrpro.operators.functionals.L1NormViewAsReal(
            weight=regularization_parameter, divide_by_n=False
        )
        stepsize = 0.97 / self.operator_norm.square()

        (sparse_code,) = mrpro.algorithms.optimizers.pgd(
            f=f,
            g=g,
            initial_value=initial_sparse_code,
            stepsize=stepsize,
            max_iterations=max_iterations,
            convergent_iterates_variant=True,
        )

        return sparse_code

    def forward(
        self,
        initial_image: torch.Tensor,
        kdata: torch.Tensor,
        mask_operator: mrpro.operators.LinearOperator,
        regularization_parameter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct image using an unrolled FISTA algorithm.

        Parameters
        ----------
        initial_image
            initial guess of the solution of the TV problem.
        kdata
            k-space data tensor of the considered problem.
        mask_operator
            mask operator that masks the k-space data.
        kernel
            convolutional kernel for the synthesis formulation.
        regularization_parameter
            regularization parameter to be used in the TV-functional. If set to None,
            it is estimated by the lambda_map_network.
            (can also be a single scalar)

        Returns
        -------
            Reconstructed image.
        """
        # if no regularization parameter map is provided, compute it with the network
        if regularization_parameter is None:
            regularization_parameter = self.parameter_map_network(
                initial_image, self.conv_dictionary_op
            )

        image_low_passed = self.low_pass_filtering_image(
            initial_image, self.low_pass_filtering_parameter
        )

        kdata_high_passed = self.high_pass_filtering_kdata(
            kdata, mask_operator @ self.fourier_operator, image_low_passed
        )

        n_filters = self.kernel.shape[0]
        initial_sparse_code = torch.zeros(
            n_filters,
            *initial_image.shape,
            device=initial_image.device,
            dtype=initial_image.dtype,
        )

        sparse_code = self.solve_sparse_coding_problem(
            initial_sparse_code,
            kdata_high_passed,
            mask_operator,
            regularization_parameter,
            max_iterations=self.n_iterations,
        )

        image = image_low_passed + self.conv_dictionary_op(sparse_code)[0]
        return image
