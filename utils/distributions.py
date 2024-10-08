import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property

__all__ = ["MultivariateLaplace", "MultivariateCauchy", "MultivariateStudentT"]


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = (
        torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
    )  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    Id = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    L = torch.linalg.solve_triangular(L_inv, Id, upper=False)
    return L















class MultivariateLaplace(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precision matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = MultivariateLaplace(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal

    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.

        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                covariance_matrix.shape[:-2], loc.shape[:-1]
            )
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
        else:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
        self.loc = loc.expand(batch_shape + (-1,))

        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateLaplace, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(MultivariateLaplace, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(
            self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return (
            self._unbroadcasted_scale_tril.pow(2)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # Ensure value is a tensor
        value = torch.as_tensor(value)

        # Compute the Mahalanobis distance
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)

        # Compute the log probability
        log_prob = -torch.norm(M, p=1, dim=-1)  # Using L1 norm for multivariate Laplace

        return log_prob


    def entropy(self):
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
        
        

        
        
        
        
        
class MultivariateCauchy(Distribution):
    r"""
    Creates a multivariate Cauchy distribution parameterized by a location vector and a scale matrix.

    Args:
        loc (Tensor): mean of the distribution
        scale_matrix (Tensor): positive-definite scale matrix

    Note:
        The scale matrix is analogous to the covariance matrix in the multivariate normal distribution.

    Example:

        >>> m = MultivariateCauchy(torch.zeros(2), torch.eye(2))
        >>> m.sample()
        tensor([0.0522, 0.3137])
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "scale_matrix": constraints.positive_definite,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, scale_matrix, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if scale_matrix.dim() < 2:
            raise ValueError("scale_matrix must be at least two-dimensional.")

        if scale_matrix.shape[-2] != scale_matrix.shape[-1]:
            raise ValueError("scale_matrix must be square.")

        batch_shape = torch.broadcast_shapes(loc.shape[:-1], scale_matrix.shape[:-2])
        self.loc = loc.expand(batch_shape + (-1,))
        self.scale_matrix = scale_matrix.expand(batch_shape + (-1, -1))
        self._unbroadcasted_scale_matrix = scale_matrix
        super().__init__(batch_shape, loc.shape[-1:], validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateCauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        scale_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new.scale_matrix = self.scale_matrix.expand(scale_shape)
        new._unbroadcasted_scale_matrix = self._unbroadcasted_scale_matrix
        super(MultivariateCauchy, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_matrix(self):
        return self._unbroadcasted_scale_matrix.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_cauchy(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_matrix, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        diff = value - self.loc
        scale_matrix_inv = torch.linalg.inv(self._unbroadcasted_scale_matrix)
        mahalanobis_sq = torch.matmul(
            torch.matmul(diff.unsqueeze(-2), scale_matrix_inv), diff.unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)

        log_prob = -self._event_shape[0] * math.log(math.pi) - torch.logdet(
            self._unbroadcasted_scale_matrix
        ) - torch.sum(torch.log1p(mahalanobis_sq), dim=-1)

        return log_prob

    def entropy(self):
        return torch.tensor(float("nan"), dtype=self.loc.dtype, device=self.loc.device)
    
    
    
    
    
    

    
class MultivariateStudentT(Distribution):
    r"""
    Creates a multivariate Student's t distribution parameterized by a location vector, a scale matrix, and degrees of freedom.

    Args:
        loc (Tensor): mean of the distribution
        scale_matrix (Tensor): positive-definite scale matrix
        df (float): degrees of freedom

    Note:
        The scale matrix is analogous to the covariance matrix in the multivariate normal distribution.

    Example:

        >>> m = MultivariateStudentT(torch.zeros(2), torch.eye(2), df=3)
        >>> m.sample()
        tensor([0.0522, 0.3137])
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "scale_matrix": constraints.positive_definite,
        "df": constraints.positive,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, scale_matrix, df, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if scale_matrix.dim() < 2:
            raise ValueError("scale_matrix must be at least two-dimensional.")

        if scale_matrix.shape[-2] != scale_matrix.shape[-1]:
            raise ValueError("scale_matrix must be square.")

        batch_shape = torch.broadcast_shapes(loc.shape[:-1], scale_matrix.shape[:-2])
        self.loc = loc.expand(batch_shape + (-1,))
        self.scale_matrix = scale_matrix.expand(batch_shape + (-1, -1))
        self.df = df
        self._unbroadcasted_scale_matrix = scale_matrix
        super().__init__(batch_shape, loc.shape[-1:], validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        scale_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new.scale_matrix = self.scale_matrix.expand(scale_shape)
        new.df = self.df
        new._unbroadcasted_scale_matrix = self._unbroadcasted_scale_matrix
        super(MultivariateStudentT, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_matrix(self):
        return self._unbroadcasted_scale_matrix.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # Draw samples from the multivariate normal distribution
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        # Draw samples from the chi-squared distribution with df degrees of freedom
        chi_sq_samples = torch.sqrt(torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device) / self.df)
        # Compute the t-distributed samples
        t_samples = self.loc + _batch_mv(self._unbroadcasted_scale_matrix, eps) / chi_sq_samples.unsqueeze(-1)
        return t_samples

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        diff = value - self.loc
        scale_matrix_inv = torch.linalg.inv(self._unbroadcasted_scale_matrix)
        mahalanobis_sq = torch.matmul(
            torch.matmul(diff.unsqueeze(-2), scale_matrix_inv), diff.unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)

        log_prob = (
            -0.5 * (self._event_shape[0] + self.df) * math.log1p(mahalanobis_sq / self.df)
            - 0.5 * torch.logdet(self._unbroadcasted_scale_matrix)
            - 0.5 * self._event_shape[0] * math.log(math.pi)
            - torch.lgamma(0.5 * self.df)
            + torch.lgamma(0.5 * (self.df + self._event_shape[0]))
        )

        return log_prob

    def entropy(self):
        half_log_det = torch.linalg.cholesky(self._unbroadcasted_scale_matrix).diagonal(dim1=-2, dim2=-1).log().sum(-1)
        H = (
            0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
            + 0.5 * (self.df + self._event_shape[0]) * (torch.digamma(0.5 * (self.df + self._event_shape[0])) - torch.digamma(0.5 * self.df))
        )
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)