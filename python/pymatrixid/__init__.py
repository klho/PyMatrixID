#*******************************************************************************
#   Copyright (C) 2013 Kenneth L. Ho
#
#   This program is free software: you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the Free
#   Software Foundation, either version 3 of the License, or (at your option)
#   any later version.
#
#   This program is distributed in the hope that it will be useful, but WITHOUT
#   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
#   more details.
#
#   You should have received a copy of the GNU General Public License along with
#   this program.  If not, see <http://www.gnu.org/licenses/>.
#*******************************************************************************

"""
Python module for interfacing with `id_dist`.
"""

from pymatrixid import backend
import numpy as np

DTYPE_ERROR   =    TypeError("invalid data type")
NARG_ERROR    =   ValueError("unknown input specification")
RETCODE_ERROR = RuntimeError("nonzero return code")

def id(*args, **kwargs):
  """
  Compute ID of a matrix.

  An ID of a matrix :math:`A` is a factorization :math:`A = BP`, where :math:`B`
  is a skeleton matrix consisting of a subset of the columns of :math:`A` and
  :math:`P` is an interpolation matrix containing the identity.

  Several call signatures are available::

    id(A, eps)
    id(A, k)
    id(m, n, matveca, eps)
    id(m, n, matveca, k)

  The first pair of signatures represents the matrix via its entries, which is
  input as the first argument. The second argument denotes either the relative
  precision of approximation for the ID or its rank, depending on its value
  (interpreted as a precision if less than or equal to one and otherwise as a
  rank). An optional argument `rand` is also available: set `rand=True` to use
  random sampling and `rand=False` to use the deterministic algorithm (default:
  `rand=True`).

  The second signature pair represents the matrix via its action on a vector.
  This is specified by the first three arguments denoting, respectively, the
  matrix row dimension, the matrix column dimension, and a function to apply the
  matrix adjoint to a vector, with call signature `y = matveca(x)`, where `x`
  and `y` are the input and output vectors, respectively. The fourth argument
  gives either the relative precision of approximation for the ID or its rank as
  described above.

  For calls specifying a precision, outputs include:

  1. The rank of the ID required to achieve this precision.
  2. A column index array specifying the ID permutation.
  3. Interpolation coefficients for the interpolation matrix in the ID.

  For calls specifying a rank, outputs include just the last two above.

  This function automatically detects the form of the call signature and the
  matrix data type, and passes the inputs to the proper backend. For details,
  see :func:`backend.iddp_id`, :func:`backend.iddp_aid`
  :func:`backend.iddp_rid`, :func:`backend.iddr_id`, :func:`backend.iddr_aid`,
  :func:`backend.iddr_rid`, :func:`backend.idzp_id`, :func:`backend.idzp_aid`,
  :func:`backend.idzp_rid`, :func:`backend.idzr_id`, :func:`backend.idzr_aid`,
  and :func:`backend.idzr_rid`.
  """
  prefix = ''
  if len(args) == 2:
    A, eps_or_k = args
    if   A.dtype ==    'float64': prefix += 'd'
    elif A.dtype == 'complex128': prefix += 'z'
    else: raise DTYPE_ERROR
    if eps_or_k < 1:
      eps = eps_or_k
      prefix += 'p'
      argstr  = 'eps, A'
    else:
      k = int(eps_or_k)
      prefix += 'r'
      argstr  = 'A, k'
    rand = kwargs.get('rand', True)
    if rand: suffix = 'a'
    else:    suffix = ''
  elif len(args) == 4:
    m, n, matveca, eps_or_k = args
    dtype = matveca(0).dtype
    if   dtype ==    'float64': prefix += 'd'
    elif dtype == 'complex128': prefix += 'z'
    else: raise DTYPE_ERROR
    if eps_or_k < 1:
      eps = eps_or_k
      prefix += 'p'
      argstr  = 'eps, m, n, matveca'
    else:
      k = int(eps_or_k)
      prefix += 'r'
      argstr  = 'm, n, matveca, k'
    suffix = 'r'
  else: raise NARG_ERROR
  s = 'backend.id%s_%sid(%s)' % (prefix, suffix, argstr)
  return eval(s)

def reconid(B, idx, proj):
  """
  Reconstruct matrix from ID.

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
  if   B.dtype ==    'float64': return backend.idd_reconid(B, idx, proj)
  elif B.dtype == 'complex128': return backend.idz_reconid(B, idx, proj)
  else: raise DTYPE_ERROR

def reconint(idx, proj):
  """
  Reconstruct interpolation matrix from ID.

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
  if   proj.dtype ==    'float64': return backend.idd_reconint(idx, proj)
  elif proj.dtype == 'complex128': return backend.idz_reconint(idx, proj)
  else: raise DTYPE_ERROR

def reconskel(A, k, idx):
  """
  Reconstruct skeleton matrix from ID.

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
  if   A.dtype ==    'float64': return backend.idd_copycols(A, k, idx)
  elif A.dtype == 'complex128': return backend.idz_copycols(A, k, idx)
  else: raise DTYPE_ERROR

def id2svd(B, idx, proj):
  """
  Convert ID to SVD.

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
  if   B.dtype ==    'float64': U, V, S = backend.idd_id2svd(B, idx, proj)
  elif B.dtype == 'complex128': U, V, S = backend.idz_id2svd(B, idx, proj)
  else: raise DTYPE_ERROR
  return U, S, V

def snorm(m, n, matvec, matveca, its=20):
  """
  Estimate spectral norm of a matrix by the randomized power method.

  This function automatically detects the matrix data type and calls the
  appropriate backend. For details, see :func:`backend.idd_snorm` and
  :func:`backend.idz_snorm`.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvec:
    Function to apply the matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function
  :param matveca:
    Function to apply the matrix adjoint to a vector, with call signature
    `y = matveca(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matveca: function
  :param its:
    Number of power method iterations.
  :type its: int

  :return:
    Spectral norm estimate.
  :rtype: float
  """
  dtype = matveca(0).dtype
  if   dtype ==    'float64':
    return backend.idd_snorm(m, n, matveca, matvec, its=its)
  elif dtype == 'complex128':
    return backend.idz_snorm(m, n, matveca, matvec, its=its)
  else: raise DTYPE_ERROR

def diffsnorm(m, n, matvec1, matvec2, matveca1, matveca2, its=20):
  """
  Estimate spectral norm of the difference of two matrices by the randomized
  power method.

  This function automatically detects the matrix data type and calls the
  appropriate backend. For details, see :func:`backend.idd_diffsnorm` and
  :func:`backend.idz_diffsnorm`.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvec1:
    Function to apply the first matrix to a vector, with call signature
    `y = matvec1(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec1: function
  :param matvec2:
    Function to apply the second matrix to a vector, with call signature
    `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec2: function
  :param matveca1:
    Function to apply the adjoint of the first matrix to a vector, with call
    signature `y = matveca1(x)`, where `x` and `y` are the input and output
    vectors, respectively.
  :type matveca1: function
  :param matveca2:
    Function to apply the adjoint of the second matrix to a vector, with call
    signature `y = matveca2(x)`, where `x` and `y` are the input and output
    vectors, respectively.
  :type matveca2: function
  :param its:
    Number of power method iterations.
  :type its: int

  :return:
    Spectral norm estimate of matrix difference.
  :rtype: float
  """
  dtype = matveca(0).dtype
  if dtype == 'float64':
    return backend.idd_diffsnorm(m, n, matveca1, matveca2, matvec1, matvec2,
                                 its=its)
  elif dtype == 'complex128':
    return backend.idz_diffsnorm(m, n, matveca1, matveca2, matvec1, matvec2,
                                 its=its)
  else: raise DTYPE_ERROR

def svd(*args, **kwargs):
  """
  Compute SVD of a matrix.

  An SVD of a matrix :math:`A` is a factorization :math:`A = U S V^{*}`, where
  :math:`U` and :math:`V` have orthonormal columns, and :math:`S` is diagonal
  with nonnegative entries.

  Several call signatures are available::

    svd(A, eps)
    svd(A, k)
    svd(m, n, matvec, matveca, eps)
    svd(m, n, matvec, matveca, k)

  The first pair of signatures represents the matrix via its entries, which is
  input as the first argument. The second argument denotes either the relative
  precision of approximation for the SVD or its rank, depending on its value
  (interpreted as a precision if less than or equal to one and otherwise as a
  rank). An optional argument `rand` is also available: set `rand=True` to use
  random sampling and `rand=False` to use the deterministic algorithm (default:
  `rand=True`).

  The second signature pair represents the matrix via its action on a vector.
  This is specified by the first four arguments denoting, respectively, the
  matrix row dimension, the matrix column dimension, a function to apply the
  matrix to a vector, and a function to apply the matrix adjoint to a vector.
  These functions have call signatures `y = matvec(x)` and `y = matveca(x)`,
  respectively, where `x` and `y` are the input and output vectors. The fifth
  argument gives either the relative precision of approximation for the SVD or
  its rank as described above.

  This function automatically detects the form of the call signature and the
  matrix data type, and passes the inputs to the proper backend. For details,
  see :func:`backend.iddp_svd`, :func:`backend.iddp_asvd`,
  :func:`backend.iddp_rsvd`, :func:`backend.iddr_svd`,
  :func:`backend.iddr_asvd`, :func:`backend.iddr_rsvd`,
  :func:`backend.idzp_svd`, :func:`backend.idzp_asvd`,
  :func:`backend.idzp_rsvd`, :func:`backend.idzr_svd`,
  :func:`backend.idzr_asvd`, :func:`backend.idzr_rsvd`.

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
  prefix = ''
  if len(args) == 2:
    A, eps_or_k = args
    if   A.dtype ==    'float64': prefix += 'd'
    elif A.dtype == 'complex128': prefix += 'z'
    else: raise DTYPE_ERROR
    if eps_or_k < 1:
      eps = eps_or_k
      prefix += 'p'
      argstr  = 'eps, A'
    else:
      k = int(eps_or_k)
      prefix += 'r'
      argstr  = 'A, k'
    rand = kwargs.get('rand', True)
    if rand: suffix = 'a'
    else:    suffix = ''
  elif len(args) == 5:
    m, n, matvec, matveca, eps_or_k = args
    dtype = matveca(0).dtype
    if   dtype ==    'float64': prefix += 'd'
    elif dtype == 'complex128': prefix += 'z'
    else: raise DTYPE_ERROR
    if eps_or_k < 1:
      eps = eps_or_k
      prefix += 'p'
      argstr  = 'eps, m, n, matveca, matvec'
    else:
      k = int(eps_or_k)
      prefix += 'r'
      argstr  = 'm, n, matveca, matvec, k'
    suffix = 'r'
  else: raise NARG_ERROR
  s = 'backend.id%s_%ssvd(%s)' % (prefix, suffix, argstr)
  U, V, S = eval(s)
  return U, S, V

def estrank(*args):
  """
  Estimate rank of a matrix to a specified relative precision using randomized
  methods.

  Several call signatures are available::

    estrank(A, eps)
    estrank(m, n, matveca, eps)

  The first signature represents the matrix via its entries, which is input as
  the first argument. The second signature represents the matrix via its action
  on a vector, which is specified by the first three arguments denoting,
  respectively, the matrix row dimension, the matrix column dimension, and a
  function to apply the matrix adjoint to a vector, with call signature
  `y = matveca(x)`, where `x` and `y` are the input and output vectors,
  respectively. In each case, the last argument gives the relative precision of
  approximation.

  If called using the first signature, the output rank is typically about 8
  higher than the actual rank.

  This function automatically detects the form of the call signature and the
  matrix data type, and passes the inputs to the proper backend. For details,
  see :func:`backend.idd_estrank`, :func:`backend.idd_findrank`,
  :func:`backend.idz_estrank`, and :func:`backend.idz_findrank`.

  :return:
    Rank estimate.
  :rtype: int
  """
  if len(args) == 2:
    A, eps = args
    if   A.dtype ==    'float64': prefix = 'd'
    elif A.dtype == 'complex128': prefix = 'z'
    else: raise DTYPE_ERROR
    suffix = 'estrank'
    argstr = 'eps, A'
  elif len(args) == 4:
    m, n, matveca, eps = args
    dtype = matveca(0).dtype
    if   dtype ==    'float64': prefix = 'd'
    elif dtype == 'complex128': prefix = 'z'
    else: raise DTYPE_ERROR
    suffix = 'findrank'
    argstr  = 'eps, m, n, matveca'
  else: raise NARG_ERROR
  s = 'backend.id%s_%s(%s)' % (prefix, suffix, argstr)
  return eval(s)