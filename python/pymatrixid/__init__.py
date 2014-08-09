#*******************************************************************************
#   Copyright (C) 2013-2014 Kenneth L. Ho
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer. Redistributions in binary
#   form must reproduce the above copyright notice, this list of conditions and
#   the following disclaimer in the documentation and/or other materials
#   provided with the distribution.
#
#   None of the names of the copyright holders may be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
#*******************************************************************************

"""
Python module for interfacing with `id_dist`.
"""

from pymatrixid import backend
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator

_DTYPE_ERROR  = TypeError("invalid data type")

def _is_real(A):
  """
  Check if matrix is real.

  :param A:
    Matrix, given as either a :class:`numpy.ndarray` or a :class:`scipy.sparse.linalg.LinearOperator`.

  :return:
    Whether matrix is real.
  :rtype: bool
  """
  try:
    if   A.dtype == np.   float64: return True
    elif A.dtype == np.complex128: return False
    else: raise _DTYPE_ERROR
  except: raise _DTYPE_ERROR

def rand(*args):
  """
  Generate standard uniform pseudorandom numbers via a very efficient lagged
  Fibonacci method.

  This routine is used for all random number generation in this package and can
  affect ID and SVD results.

  Several call signatures are available:

  - If no arguments are given, then the seed values are reset to their original
    values.

  - If an integer `n` is given as input, then an array of `n` pseudorandom
    numbers are returned.

  - If an array `s` of 55 values is given as input, then the seed values are
    set to `s`.

  For details, see :func:`backend.id_srand`, :func:`backend.id_srandi`, and
  :func:`backend.id_srando`.
  """
  if   len(args) == 0: backend.id_srando()
  elif len(args) == 1:
    x = np.asfortranarray(args[0])
    if   x.size ==  1: return backend.id_srand (x)
    elif x.size == 55:        backend.id_srandi(x)
    else: raise ValueError("invalid input size")
  else: raise ValueError("unknown input specification")

def interp_decomp(A, eps_or_k, rand=True):
  """
  Compute ID of a matrix.

  An ID of a matrix `A` is a factorization defined by a rank `k`, a column index
  array `idx`, and interpolation coefficients `proj` such that::

    numpy.dot(A[:,idx[:k]], proj) = A[:,idx[k:]]

  The original matrix can then be reconstructed as::

    numpy.hstack([A[:,idx[:k]],
                  numpy.dot(A[:,idx[:k]], proj)]
                )[:,numpy.argsort(idx)]

  or via the routine :func:`reconstruct_matrix_from_id`. This can equivalently
  be written as::

    numpy.dot(A[:,idx[:k]],
              numpy.hstack([numpy.eye(k), proj])
             )[:,np.argsort(idx)]

  in terms of the skeleton and interpolation matrices::

    B = A[:,idx[:k]]

  and::

    P = numpy.hstack([numpy.eye(k), proj])[:,np.argsort(idx)]

  respectively. See also :func:`reconstruct_interp_matrix` and
  :func:`reconstruct_skel_matrix`.

  The ID can be computed to any relative precision or rank (depending on the
  value of `eps_or_k`). If a precision is specified (`eps_or_k < 1`), then this
  function has the output signature::

    k, idx, proj = interp_decomp(A, eps_or_k)

  Otherwise, if a rank is specified (`eps_or_k >= 1`), then the output signature
  is::

    idx, proj = interp_decomp(A, eps_or_k)

  This function automatically detects the form of the input parameters and
  passes them to the appropriate backend. For details, see
  :func:`backend.iddp_id`, :func:`backend.iddp_aid`, :func:`backend.iddp_rid`,
  :func:`backend.iddr_id`, :func:`backend.iddr_aid`, :func:`backend.iddr_rid`,
  :func:`backend.idzp_id`, :func:`backend.idzp_aid`, :func:`backend.idzp_rid`,
  :func:`backend.idzr_id`, :func:`backend.idzr_aid`, and
  :func:`backend.idzr_rid`.

  :param A:
    Matrix to be factored, given as either a :class:`numpy.ndarray` or a
    :class:`scipy.sparse.linalg.LinearOperator` with the `rmatvec` method (to
    apply the matrix adjoint).
  :type A: :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
  :param eps_or_k:
    Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
    approximation.
  :type eps_or_k: float or int
  :keyword rand:
    Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
    (randomized algorithms are always used if `A` is of type
    :class:`scipy.sparse.linalg.LinearOperator`).
  :type rand: bool

  :return:
    Rank required to achieve specified relative precision if
    `eps_or_k < 1`.
  :rtype: int
  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  real = _is_real(A)
  prec = eps_or_k < 1
  if prec: eps = eps_or_k
  else:    k = int(eps_or_k)
  if isinstance(A, np.ndarray):
    if prec:
      if rand:
        if real: k, idx, proj = backend.iddp_aid(eps, A)
        else:    k, idx, proj = backend.idzp_aid(eps, A)
      else:
        if real: k, idx, proj = backend.iddp_id(eps, A)
        else:    k, idx, proj = backend.idzp_id(eps, A)
      return k, idx - 1, proj
    else:
      if rand:
        if real: idx, proj = backend.iddr_aid(A, k)
        else:    idx, proj = backend.idzr_aid(A, k)
      else:
        if real: idx, proj = backend.iddr_id(A, k)
        else:    idx, proj = backend.idzr_id(A, k)
      return idx - 1, proj
  elif isinstance(A, LinearOperator):
    m, n = A.shape
    matveca = A.rmatvec
    if prec:
      if real: k, idx, proj = backend.iddp_rid(eps, m, n, matveca)
      else:    k, idx, proj = backend.idzp_rid(eps, m, n, matveca)
      return k, idx - 1, proj
    else:
      if real: idx, proj = backend.iddr_rid(m, n, matveca, k)
      else:    idx, proj = backend.idzr_rid(m, n, matveca, k)
      return idx - 1, proj
  else: raise _DTYPE_ERROR

def reconstruct_matrix_from_id(B, idx, proj):
  """
  Reconstruct matrix from its ID.

  A matrix `A` with skeleton matrix `B` and ID indices and coefficients `idx`
  and `proj`, respectively, can be reconstructed as::

    numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

  See also :func:`reconstruct_interp_matrix` and
  :func:`reconstruct_skel_matrix`.

  This function automatically detects the matrix data type and calls the
  appropriate backend. For details, see :func:`backend.idd_reconid` and
  :func:`backend.idz_reconid`.

  :param B:
    Skeleton matrix.
  :type B: :class:`numpy.ndarray`
  :param idx:
    Column index array.
  :type idx: :class:`numpy.ndarray`
  :param proj:
    Interpolation coefficients.
  :type proj: :class:`numpy.ndarray`

  :return:
    Reconstructed matrix.
  :rtype: :class:`numpy.ndarray`
  """
  if   B.dtype ==    'float64': return backend.idd_reconid(B, idx + 1, proj)
  elif B.dtype == 'complex128': return backend.idz_reconid(B, idx + 1, proj)
  else: raise _DTYPE_ERROR

def reconstruct_interp_matrix(idx, proj):
  """
  Reconstruct interpolation matrix from ID.

  The interpolation matrix can be reconstructed from the ID indices and
  coefficients `idx` and `proj`, respectively, as::

    P = numpy.hstack([numpy.eye(proj.shape[0]), proj])[:,numpy.argsort(idx)]

  The original matrix can then be reconstructed from its skeleton matrix `B`
  via::

    numpy.dot(B, P)

  See also :func:`reconstruct_matrix_from_id` and
  :func:`reconstruct_skel_matrix`.

  This function automatically detects the matrix data type and calls the
  appropriate backend. For details, see :func:`backend.idd_reconint` and
  :func:`backend.idz_reconint`.

  :param idx:
    Column index array.
  :type idx: :class:`numpy.ndarray`
  :param proj:
    Interpolation coefficients.
  :type proj: :class:`numpy.ndarray`

  :return:
    Interpolation matrix.
  :rtype: :class:`numpy.ndarray`
  """
  if   proj.dtype ==    'float64': return backend.idd_reconint(idx + 1, proj)
  elif proj.dtype == 'complex128': return backend.idz_reconint(idx + 1, proj)
  else: raise _DTYPE_ERROR

def reconstruct_skel_matrix(A, k, idx):
  """
  Reconstruct skeleton matrix from ID.

  The skeleton matrix can be reconstructed from the original matrix `A` and its
  ID rank and indices `k` and `idx`, respectively, as::

    B = A[:,idx[:k]]

  The original matrix can then be reconstructed via::

    numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

  See also :func:`reconstruct_matrix_from_id` and
  :func:`reconstruct_interp_matrix`.

  This function automatically detects the matrix data type and calls the
  appropriate backend. For details, see :func:`backend.idd_copycols` and
  :func:`backend.idz_copycols`.

  :param A:
    Original matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of ID.
  :type k: int
  :param idx:
    Column index array.
  :type idx: :class:`numpy.ndarray`

  :return:
    Skeleton matrix.
  :rtype: :class:`numpy.ndarray`
  """
  if   A.dtype ==    'float64': return backend.idd_copycols(A, k, idx + 1)
  elif A.dtype == 'complex128': return backend.idz_copycols(A, k, idx + 1)
  else: raise _DTYPE_ERROR

def id_to_svd(B, idx, proj):
  """
  Convert ID to SVD.

  The SVD reconstruction of a matrix with skeleton matrix `B` and ID indices and
  coefficients `idx` and `proj`, respectively, is::

    U, S, V = id_to_svd(B, idx, proj)
    A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))

  See also :func:`svd`.

  This function automatically detects the matrix data type and calls the
  appropriate backend. For details, see :func:`backend.idd_id2svd` and
  :func:`backend.idz_id2svd`.

  :param B:
    Skeleton matrix.
  :type B: :class:`numpy.ndarray`
  :param idx:
    Column index array.
  :type idx: :class:`numpy.ndarray`
  :param proj:
    Interpolation coefficients.
  :type proj: :class:`numpy.ndarray`

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  """
  if   B.dtype ==    'float64': U, V, S = backend.idd_id2svd(B, idx + 1, proj)
  elif B.dtype == 'complex128': U, V, S = backend.idz_id2svd(B, idx + 1, proj)
  else: raise _DTYPE_ERROR
  return U, S, V

def estimate_spectral_norm(A, its=20):
  """
  Estimate spectral norm of a matrix by the randomized power method.

  This function automatically detects the matrix data type and calls the
  appropriate backend. For details, see :func:`backend.idd_snorm` and
  :func:`backend.idz_snorm`.

  :param A:
    Matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
    `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
  :type A: :class:`scipy.sparse.linalg.LinearOperator`
  :keyword its:
    Number of power method iterations.
  :type its: int

  :return:
    Spectral norm estimate.
  :rtype: float
  """
  A = aslinearoperator(A)
  m, n = A.shape
  matvec  = lambda x: A. matvec(x)
  matveca = lambda x: A.rmatvec(x)
  if A.dtype == 'float64':
    return backend.idd_snorm(m, n, matveca, matvec, its=its)
  elif A.dtype == 'complex128':
    return backend.idz_snorm(m, n, matveca, matvec, its=its)
  else: raise _DTYPE_ERROR

def estimate_spectral_norm_diff(A, B, its=20):
  """
  Estimate spectral norm of the difference of two matrices by the randomized
  power method.

  This function automatically detects the matrix data type and calls the
  appropriate backend. For details, see :func:`backend.idd_diffsnorm` and
  :func:`backend.idz_diffsnorm`.

  :param A:
    First matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
    `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
  :type A: :class:`scipy.sparse.linalg.LinearOperator`
  :param B:
    Second matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with
    the `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
  :type B: :class:`scipy.sparse.linalg.LinearOperator`
  :keyword its:
    Number of power method iterations.
  :type its: int

  :return:
    Spectral norm estimate of matrix difference.
  :rtype: float
  """
  A = aslinearoperator(A)
  B = aslinearoperator(B)
  m, n = A.shape
  matvec1  = lambda x: A. matvec(x)
  matveca1 = lambda x: A.rmatvec(x)
  matvec2  = lambda x: B. matvec(x)
  matveca2 = lambda x: B.rmatvec(x)
  if A.dtype == 'float64':
    return backend.idd_diffsnorm(m, n, matveca1, matveca2, matvec1, matvec2,
                                 its=its)
  elif A.dtype == 'complex128':
    return backend.idz_diffsnorm(m, n, matveca1, matveca2, matvec1, matvec2,
                                 its=its)
  else: raise _DTYPE_ERROR

def svd(A, eps_or_k, rand=True):
  """
  Compute SVD of a matrix via an ID.

  An SVD of a matrix `A` is a factorization::

    A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))

  where `U` and `V` have orthonormal columns and `S` is nonnegative.

  The SVD can be computed to any relative precision or rank (depending on the
  value of `eps_or_k`).

  See also :func:`interp_decomp` and :func:`id_to_svd`.

  This function automatically detects the form of the input parameters and
  passes them to the appropriate backend. For details, see
  :func:`backend.iddp_svd`, :func:`backend.iddp_asvd`,
  :func:`backend.iddp_rsvd`, :func:`backend.iddr_svd`,
  :func:`backend.iddr_asvd`, :func:`backend.iddr_rsvd`,
  :func:`backend.idzp_svd`, :func:`backend.idzp_asvd`,
  :func:`backend.idzp_rsvd`, :func:`backend.idzr_svd`,
  :func:`backend.idzr_asvd`, and :func:`backend.idzr_rsvd`.

  :param A:
    Matrix to be factored, given as either a :class:`numpy.ndarray` or a
    :class:`scipy.sparse.linalg.LinearOperator` with the `matvec` and `rmatvec`
    methods (to apply the matrix and its adjoint).
  :type A: :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
  :param eps_or_k:
    Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
    approximation.
  :type eps_or_k: float or int
  :keyword rand:
    Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
    (randomized algorithms are always used if `A` is of type
    :class:`scipy.sparse.linalg.LinearOperator`).
  :type rand: bool

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  """
  real = _is_real(A)
  prec = eps_or_k < 1
  if prec: eps = eps_or_k
  else:    k = int(eps_or_k)
  if isinstance(A, np.ndarray):
    if prec:
      if rand:
        if real: U, V, S = backend.iddp_asvd(eps, A)
        else:    U, V, S = backend.idzp_asvd(eps, A)
      else:
        if real: U, V, S = backend.iddp_svd(eps, A)
        else:    U, V, S = backend.idzp_svd(eps, A)
    else:
      if rand:
        if real: U, V, S = backend.iddr_asvd(A, k)
        else:    U, V, S = backend.idzr_asvd(A, k)
      else:
        if real: U, V, S = backend.iddr_svd(A, k)
        else:    U, V, S = backend.idzr_svd(A, k)
  elif isinstance(A, LinearOperator):
    m, n = A.shape
    matvec  = lambda x: A.matvec (x)
    matveca = lambda x: A.rmatvec(x)
    if prec:
      if real: U, V, S = backend.iddp_rsvd(eps, m, n, matveca, matvec)
      else:    U, V, S = backend.idzp_rsvd(eps, m, n, matveca, matvec)
    else:
      if real: U, V, S = backend.iddr_rsvd(m, n, matveca, matvec, k)
      else:    U, V, S = backend.idzr_rsvd(m, n, matveca, matvec, k)
  else: raise _DTYPE_ERROR
  return U, S, V

def estimate_rank(A, eps):
  """
  Estimate matrix rank to a specified relative precision using randomized
  methods.

  The matrix `A` can be given as either a :class:`numpy.ndarray` or a
  :class:`scipy.sparse.linalg.LinearOperator`, with different algorithms used
  for each case. If `A` is of type :class:`numpy.ndarray`, then the output rank
  is typically about 8 higher than the actual numerical rank.

  This function automatically detects the form of the input parameters and
  passes them to the appropriate backend. For details,
  see :func:`backend.idd_estrank`, :func:`backend.idd_findrank`,
  :func:`backend.idz_estrank`, and :func:`backend.idz_findrank`.

  :param A:
    Matrix whose rank is to be estimated, given as either a
    :class:`numpy.ndarray` or a :class:`scipy.sparse.linalg.LinearOperator` with
    the `rmatvec` method (to apply the matrix adjoint).
  :type A: :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
  :param eps:
    Relative error for numerical rank definition.
  :type eps: float

  :return:
    Estimated matrix rank.
  :rtype: int
  """
  real = _is_real(A)
  if isinstance(A, np.ndarray):
    if real: rank = backend.idd_estrank(eps, A)
    else:    rank = backend.idz_estrank(eps, A)
    if rank == 0: rank = min(A.shape)
    return rank
  elif isinstance(A, LinearOperator):
    m, n = A.shape
    matveca = A.rmatvec
    if real: return backend.idd_findrank(eps, m, n, matveca)
    else:    return backend.idz_findrank(eps, m, n, matveca)
  else: raise _DTYPE_ERROR