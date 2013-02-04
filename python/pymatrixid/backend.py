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
Direct wrappers for Fortran `id_dist` backend.
"""

import id_dist as _id
import numpy as np

#-------------------------------------------------------------------------------
# idd_frm.f
#-------------------------------------------------------------------------------

def idd_frm(n, w, x):
  """
  Transform real vector via a composition of Rokhlin's random transform, random
  subselection, and an FFT.

  In contrast to :func:`idd_sfrm`, this routine works best when the length of
  the transformed vector is the power-of-two integer output by :func:`idd_frmi`,
  or when the length is not specified but instead determined a posteriori from
  the output. The returned transformed vector is randomly permuted.

  :param n:
    Greatest power-of-two integer satisfying `n <= x.size` as obtained from
    :func:`idd_frmi`; `n` is also the length of the output vector.
  :type n: int
  :param w:
    Initialization array constructed by :func:`idd_frmi`.
  :type w: :class:`numpy.ndarray`
  :param x:
    Vector to be transformed.
  :type x: :class:`numpy.ndarray`

  :return:
    Transformed vector.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idd_frm(n, w, x)

def idd_sfrm(l, n, w, x):
  """
  Transform real vector via a composition of Rokhlin's random transform, random subselection, and an FFT.

  In contrast to :func:`idd_frm`, this routine works best when the length of the
  transformed vector is known a priori.

  :param l:
    Length of transformed vector, satisfying `l <= n`.
  :type l: int
  :param n:
    Greatest power-of-two integer satisfying `n <= x.size` as obtained from
    :func:`idd_sfrmi`.
  :type n: int
  :param w:
    Initialization array constructed by :func:`idd_sfrmi`.
  :type w: :class:`numpy.ndarray`
  :param x:
    Vector to be transformed.
  :type x: :class:`numpy.ndarray`

  :return:
    Transformed vector.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idd_sfrm(l, n, w, x)

def idd_frmi(m):
  """
  Initialize data for :func:`idd_frm`.

  :param m:
    Length of vector to be transformed.
  :type m: int

  :return:
    Greatest power-of-two integer `n` satisfying `n <= m`.
  :rtype: int
  :return:
    Initialization array to be used by :func:`idd_frm`.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idd_frmi(m)

def idd_sfrmi(l, m):
  """
  Initialize data for :func:`idd_sfrm`.

  :param l:
    Length of output transformed vector.
  :type l: int
  :param m:
    Length of the vector to be transformed.
  :type m: int

  :return:
    Greatest power-of-two integer `n` satisfying `n <= m`.
  :rtype: int
  :return:
    Initialization array to be used by :func:`idd_sfrm`.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idd_sfrmi(l, m)

#-------------------------------------------------------------------------------
# idd_id.f
#-------------------------------------------------------------------------------

def iddp_id(eps, A):
  """
  Compute ID of a real matrix to a specified relative precision.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Rank of ID.
  :rtype: int
  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=True, dtype='float64', order='F')
  k, idx, rnorms = _id.iddp_id(eps, A)
  n = A.shape[1]
  A = A.T.reshape(A.size)
  proj = A[:k*(n-k)].reshape((k, n-k), order='F')
  return k, idx, proj

def iddr_id(A, k):
  """
  Compute ID of a real matrix to a specified rank.

  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of ID.
  :type k: int

  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=True, dtype='float64', order='F')
  idx, rnorms = _id.iddr_id(A, k)
  n = A.shape[1]
  A = A.T.reshape(A.size)
  proj = A[:k*(n-k)].reshape((k, n-k), order='F')
  return idx, proj

def idd_reconid(B, idx, proj):
  """
  Reconstruct matrix from real ID.

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
  B = np.array(B, copy=False, dtype='float64', order='F')
  if proj.size > 0:
    return _id.idd_reconid(B, idx, proj)
  else:
    P = idd_reconint(idx, proj)
    return np.dot(B, P)

def idd_reconint(idx, proj):
  """
  Reconstruct interpolation matrix from real ID.

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
  return _id.idd_reconint(idx, proj)

def idd_copycols(A, k, idx):
  """
  Reconstruct skeleton matrix from real ID.

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
  A = np.array(A, copy=False, dtype='float64', order='F')
  return _id.idd_copycols(A, k, idx)

#-------------------------------------------------------------------------------
# idd_id2svd.f
#-------------------------------------------------------------------------------

def idd_id2svd(B, idx, proj):
  """
  Convert real ID to SVD.

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
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  B = np.array(B, copy=False, dtype='float64', order='F')
  U, V, S, ier = _id.idd_id2svd(B, idx, proj)
  if ier != 0: raise RETCODE_ERROR
  return U, V, S

#-------------------------------------------------------------------------------
# idd_snorm.f
#-------------------------------------------------------------------------------

def idd_snorm(m, n, matvect, matvec, its=20):
  """
  Estimate spectral norm of a real matrix by the randomized power method.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvect:
    Function to apply the matrix transpose to a vector, with call signature
    `y = matvect(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvect: function
  :param matvec:
    Function to apply the matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function
  :param its:
    Number of power method iterations.
  :type its: int

  :return:
    Spectral norm estimate.
  :rtype: float
  """
  snorm, v = _id.idd_snorm(m, n, matvect, matvec, its)
  return snorm

def idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its=20):
  """
  Estimate spectral norm of the difference of two real matrices by the
  randomized power method.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvect:
    Function to apply the transpose of the first matrix to a vector, with call
    signature `y = matvect(x)`, where `x` and `y` are the input and output
    vectors, respectively.
  :type matvect: function
  :param matvect2:
    Function to apply the transpose of the second matrix to a vector, with call
    signature `y = matvect2(x)`, where `x` and `y` are the input and output
    vectors, respectively.
  :type matvect2: function
  :param matvec:
    Function to apply the first matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function
  :param matvec2:
    Function to apply the second matrix to a vector, with call signature
    `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec2: function
  :param its:
    Number of power method iterations.
  :type its: int

  :return:
    Spectral norm estimate of matrix difference.
  :rtype: float
  """
  return _id.idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its)

#-------------------------------------------------------------------------------
# idd_svd.f
#-------------------------------------------------------------------------------

def iddr_svd(A, k):
  """
  Compute SVD of a real matrix to a specified rank.

  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of SVD.
  :type k: int

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=True, dtype='float64', order='F')
  U, V, S, ier = _id.iddr_svd(A, k)
  if ier != 0: raise RETCODE_ERROR
  return U, V, S

def iddp_svd(eps, A):
  """
  Compute SVD of a real matrix to a specified relative precision.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=True, dtype='float64', order='F')
  m, n = A.shape
  k, iU, iV, iS, w, ier = _id.iddp_svd(eps, A)
  if ier != 0: raise RETCODE_ERROR
  U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
  V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
  S = w[iS-1:iS+k-1]
  return U, V, S

#-------------------------------------------------------------------------------
# iddp_aid.f
#-------------------------------------------------------------------------------

def iddp_aid(eps, A):
  """
  Compute ID of a real matrix to a specified relative precision using random
  sampling.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Rank of ID.
  :rtype: int
  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=False, dtype='float64', order='F')
  m, n = A.shape
  n2, w = idd_frmi(m)
  proj = np.empty(n*(2*n2 + 1) + n2 + 1)
  k, idx, proj = _id.iddp_aid(eps, A, w, proj)
  proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
  return k, idx, proj

def idd_estrank(eps, A):
  """
  Estimate rank of a real matrix to a specified relative precision using random
  sampling.

  The output rank is typically about 8 higher than the actual rank.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Rank estimate.
  :rtype: int
  """
  A = np.array(A, copy=False, dtype='float64', order='F')
  m, n = A.shape
  n2, w = idd_frmi(m)
  ra = np.empty(n*n2 + (n + 1)*(n2 + 1))
  k, ra = _id.idd_estrank(eps, A, w, ra)
  return k

#-------------------------------------------------------------------------------
# iddp_asvd.f
#-------------------------------------------------------------------------------

def iddp_asvd(eps, A):
  """
  Compute SVD of a real matrix to a specified relative precision using random
  sampling.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=False, dtype='float64', order='F')
  m, n = A.shape
  n2, winit = _id.idd_frmi(m)
  w = np.empty(max((min(m,n) + 1)*(3*m + 5*n + 1) + 25*min(m,n)**2,
                   (2*n + 1)*(n2 + 1)))
  k, iU, iV, iS, w, ier = _id.iddp_asvd(eps, A, winit, w)
  U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
  V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
  S = w[iS-1:iS+k-1]
  return U, V, S

#-------------------------------------------------------------------------------
# iddp_rid.f
#-------------------------------------------------------------------------------

def iddp_rid(eps, m, n, matvect):
  """
  Compute ID of a real matrix to a specified relative precision using random
  matrix-vector multiplication.

  :param eps:
    Relative precision.
  :type eps: float
  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvect:
    Function to apply the matrix transpose to a vector, with call signature
    `y = matvect(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvect: function

  :return:
    Rank of ID.
  :rtype: int
  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  proj = np.empty(m + 1 + 2*n*(min(m,n) + 1))
  k, idx, proj, ier = _id.iddp_rid(eps, m, n, matvect, proj)
  if ier != 0: raise RETCODE_ERROR
  proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
  return k, idx, proj

def idd_findrank(eps, m, n, matvect):
  """
  Estimate rank of a real matrix to a specified relative precision using random
  matrix-vector multiplication.

  :param eps:
    Relative precision.
  :type eps: float
  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvect:
    Function to apply the matrix transpose to a vector, with call signature
    `y = matvect(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvect: function

  :return:
    Rank estimate.
  :rtype: int
  """
  k, ra, ier = _id.idd_findrank(eps, m, n, matvect)
  if ier != 0: raise RETCODE_ERROR
  return k

#-------------------------------------------------------------------------------
# iddp_rsvd.f
#-------------------------------------------------------------------------------

def iddp_rsvd(eps, m, n, matvect, matvec):
  """
  Compute SVD of a real matrix to a specified relative precision using random
  matrix-vector multiplication.

  :param eps:
    Relative precision.
  :type eps: float
  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvect:
    Function to apply the matrix transpose to a vector, with call signature
    `y = matvect(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvect: function
  :param matvec:
    Function to apply the matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  k, iU, iV, iS, w, ier = _id.iddp_rsvd(eps, m, n, matvect, matvec)
  if ier != 0: raise RETCODE_ERROR
  U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
  V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
  S = w[iS-1:iS+k-1]
  return U, V, S

#-------------------------------------------------------------------------------
# iddr_aid.f
#-------------------------------------------------------------------------------

def iddr_aid(A, k):
  """
  Compute ID of a real matrix to a specified rank using random sampling.

  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of ID.
  :type k: int

  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=False, dtype='float64', order='F')
  m, n = A.shape
  w = iddr_aidi(m, n, k)
  idx, proj = _id.iddr_aid(A, k, w)
  if k == n: proj = np.array([], dtype='float64')
  proj = proj.reshape((k, n-k), order='F')
  return idx, proj

def iddr_aidi(m, n, k):
  """
  Initialize array for :func:`iddr_aid`.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param k:
    Rank of ID.
  :type k: int

  :return:
    Initialization array to be used by :func:`iddr_aid`.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.iddr_aidi(m, n, k)

#-------------------------------------------------------------------------------
# iddr_asvd.f
#-------------------------------------------------------------------------------

def iddr_asvd(A, k):
  """
  Compute SVD of a real matrix to a specified rank using random sampling.

  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of SVD.
  :type k: int

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=False, dtype='float64', order='F')
  m, n = A.shape
  w = np.empty((2*k + 28)*m + (6*k + 21)*n + 25*k**2 + 100)
  w_ = iddr_aidi(m, n, k)
  w[:w_.size] = w_
  U, V, S, ier = _id.iddr_asvd(A, k, w)
  if ier != 0: raise RETCODE_ERROR
  return U, V, S

#-------------------------------------------------------------------------------
# iddr_rid.f
#-------------------------------------------------------------------------------

def iddr_rid(m, n, matvect, k):
  """
  Compute ID of a real matrix to a specified rank using random matrix-vector
  multiplication.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvect:
    Function to apply the matrix transpose to a vector, with call signature
    `y = matvect(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvect: function
  :param k:
    Rank of ID.
  :type k: int

  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  idx, proj = _id.iddr_rid(m, n, matvect, k)
  proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
  return idx, proj

#-------------------------------------------------------------------------------
# iddr_rsvd.f
#-------------------------------------------------------------------------------

def iddr_rsvd(m, n, matvect, matvec, k):
  """
  Compute SVD of a real matrix to a specified rank using random matrix-vector
  multiplication.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matvect:
    Function to apply the matrix transpose to a vector, with call signature
    `y = matvect(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvect: function
  :param matvec:
    Function to apply the matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function
  :param k:
    Rank of SVD.
  :type k: int

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  U, V, S, ier = _id.iddr_rsvd(m, n, matvect, matvec, k)
  if ier != 0: raise RETCODE_ERROR
  return U, V, S

#-------------------------------------------------------------------------------
# idz_frm.f
#-------------------------------------------------------------------------------

def idz_frm(n, w, x):
  """
  Transform complex vector via a composition of Rokhlin's random transform,
  random subselection, and an FFT.

  In contrast to :func:`idz_sfrm`, this routine works best when the length of
  the transformed vector is the power-of-two integer output by :func:`idz_frmi`,
  or when the length is not specified but instead determined a posteriori from
  the output. The returned transformed vector is randomly permuted.

  :param n:
    Greatest power-of-two integer satisfying `n <= x.size` as obtained from
    :func:`idz_frmi`; `n` is also the length of the output vector.
  :type n: int
  :param w:
    Initialization array constructed by :func:`idz_frmi`.
  :type w: :class:`numpy.ndarray`
  :param x:
    Vector to be transformed.
  :type x: :class:`numpy.ndarray`

  :return:
    Transformed vector.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idz_frm(n, w, x)

def idz_sfrm(l, n, w, x):
  """
  Transform complex vector via a composition of Rokhlin's random transform,
  random subselection, and an FFT.

  In contrast to :func:`idz_frm`, this routine works best when the length of the
  transformed vector is known a priori.

  :param l:
    Length of transformed vector, satisfying `l <= n`.
  :type l: int
  :param n:
    Greatest power-of-two integer satisfying `n <= x.size` as obtained from
    :func:`idz_sfrmi`.
  :type n: int
  :param w:
    Initialization array constructed by :func:`idd_sfrmi`.
  :type w: :class:`numpy.ndarray`
  :param x:
    Vector to be transformed.
  :type x: :class:`numpy.ndarray`

  :return:
    Transformed vector.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idz_sfrm(l, n, w, x)

def idz_frmi(m):
  """
  Initialize data for :func:`idz_frm`.

  :param m:
    Length of vector to be transformed.
  :type m: int

  :return:
    Greatest power-of-two integer `n` satisfying `n <= m`.
  :rtype: int
  :return:
    Initialization array to be used by :func:`idz_frm`.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idz_frmi(m)

def idz_sfrmi(l, m):
  """
  Initialize data for :func:`idz_sfrm`.

  :param l:
    Length of output transformed vector.
  :type l: int
  :param m:
    Length of the vector to be transformed.
  :type m: int

  :return:
    Greatest power-of-two integer `n` satisfying `n <= m`.
  :rtype: int
  :return:
    Initialization array to be used by :func:`idz_sfrm`.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idz_sfrmi(l, m)

#-------------------------------------------------------------------------------
# idz_id.f
#-------------------------------------------------------------------------------

def idzp_id(eps, A):
  """
  Compute ID of a complex matrix to a specified relative precision.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Rank of ID.
  :rtype: int
  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=True, dtype='complex128', order='F')
  k, idx, rnorms = _id.idzp_id(eps, A)
  n = A.shape[1]
  A = A.T.reshape(A.size)
  proj = A[:k*(n-k)].reshape((k, n-k), order='F')
  return k, idx, proj

def idzr_id(A, k):
  """
  Compute ID of a complex matrix to a specified rank.

  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of ID.
  :type k: int

  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=True, dtype='complex128', order='F')
  idx, rnorms = _id.idzr_id(A, k)
  n = A.shape[1]
  A = A.T.reshape(A.size)
  proj = A[:k*(n-k)].reshape((k, n-k), order='F')
  return idx, proj

def idz_reconid(B, idx, proj):
  """
  Reconstruct matrix from complex ID.

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
  B = np.array(B, copy=False, dtype='complex128', order='F')
  if proj.size > 0:
    return _id.idd_reconid(B, idx, proj)
  else:
    P = idd_reconint(idx, proj)
    return np.dot(B, P)

def idz_reconint(idx, proj):
  """
  Reconstruct interpolation matrix from complex ID.

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
  return _id.idz_reconint(idx, proj)

def idz_copycols(A, k, idx):
  """
  Reconstruct skeleton matrix from complex ID.

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
  A = np.array(A, copy=False, dtype='complex128', order='F')
  return _id.idz_copycols(A, k, idx)

#-------------------------------------------------------------------------------
# idz_id2svd.f
#-------------------------------------------------------------------------------

def idz_id2svd(B, idx, proj):
  """
  Convert complex ID to SVD.

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
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  B = np.array(B, copy=False, dtype='complex128', order='F')
  U, V, S, ier = _id.idz_id2svd(B, idx, proj)
  if ier != 0: raise RETCODE_ERROR
  return U, V, S

#-------------------------------------------------------------------------------
# idz_snorm.f
#-------------------------------------------------------------------------------

def idz_snorm(m, n, matveca, matvec, its=20):
  """
  Estimate spectral norm of a complex matrix by the randomized power method.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matveca:
    Function to apply the matrix adjoint to a vector, with call signature
    `y = matveca(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matveca: function
  :param matvec:
    Function to apply the matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function
  :param its:
    Number of power method iterations.
  :type its: int

  :return:
    Spectral norm estimate.
  :rtype: float
  """
  snorm, v = _id.idz_snorm(m, n, matveca, matvec, its)
  return snorm

def idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its=20):
  """
  Estimate spectral norm of the difference of two complex matrices by the
  randomized power method.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matveca:
    Function to apply the adjoint of the first matrix to a vector, with call
    signature `y = matveca(x)`, where `x` and `y` are the input and output
    vectors, respectively.
  :type matveca: function
  :param matveca2:
    Function to apply the adjoint of the second matrix to a vector, with call
    signature `y = matveca2(x)`, where `x` and `y` are the input and output
    vectors, respectively.
  :type matveca2: function
  :param matvec:
    Function to apply the first matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function
  :param matvec2:
    Function to apply the second matrix to a vector, with call signature
    `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec2: function
  :param its:
    Number of power method iterations.
  :type its: int

  :return:
    Spectral norm estimate of matrix difference.
  :rtype: float
  """
  return _id.idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its)

#-------------------------------------------------------------------------------
# idz_svd.f
#-------------------------------------------------------------------------------

def idzr_svd(A, k):
  """
  Compute SVD of a complex matrix to a specified rank.

  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of SVD.
  :type k: int

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=True, dtype='complex128', order='F')
  U, V, S, ier = _id.idzr_svd(A, k)
  if ier != 0: raise RETCODE_ERROR
  return U, V, S

def idzp_svd(eps, A):
  """
  Compute SVD of a complex matrix to a specified relative precision.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=True, dtype='complex128', order='F')
  m, n = A.shape
  k, iU, iV, iS, w, ier = _id.idzp_svd(eps, A)
  if ier != 0: raise RETCODE_ERROR
  U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
  V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
  S = w[iS-1:iS+k-1]
  return U, V, S

#-------------------------------------------------------------------------------
# idzp_aid.f
#-------------------------------------------------------------------------------

def idzp_aid(eps, A):
  """
  Compute ID of a complex matrix to a specified relative precision using random
  sampling.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Rank of ID.
  :rtype: int
  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=False, dtype='complex128', order='F')
  m, n = A.shape
  n2, w = idz_frmi(m)
  proj = np.empty(n*(2*n2 + 1) + n2 + 1, dtype='complex128')
  k, idx, proj = _id.idzp_aid(eps, A, w, proj)
  proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
  return k, idx, proj

def idz_estrank(eps, A):
  """
  Estimate rank of a complex matrix to a specified relative precision using
  random sampling.

  The output rank is typically about 8 higher than the actual rank.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Rank estimate.
  :rtype: int
  """
  A = np.array(A, copy=False, dtype='complex128', order='F')
  m, n = A.shape
  n2, w = idz_frmi(m)
  ra = np.empty(n*n2 + (n + 1)*(n2 + 1), dtype='complex128')
  k, ra = _id.idz_estrank(eps, A, w, ra)
  return k

#-------------------------------------------------------------------------------
# idzp_asvd.f
#-------------------------------------------------------------------------------

def idzp_asvd(eps, A):
  """
  Compute SVD of a complex matrix to a specified relative precision using random
  sampling.

  :param eps:
    Relative precision.
  :type eps: float
  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=False, dtype='complex128', order='F')
  m, n = A.shape
  n2, winit = _id.idz_frmi(m)
  w = np.empty(max((min(m,n) + 1)*(3*m + 5*n + 11) + 8*min(m,n)**2,
                   (2*n + 1)*(n2 + 1)),
               dtype='complex128')
  k, iU, iV, iS, w, ier = _id.idzp_asvd(eps, A, winit, w)
  U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
  V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
  S = w[iS-1:iS+k-1]
  return U, V, S

#-------------------------------------------------------------------------------
# idzp_rid.f
#-------------------------------------------------------------------------------

def idzp_rid(eps, m, n, matveca):
  """
  Compute ID of a complex matrix to a specified relative precision using random
  matrix-vector multiplication.

  :param eps:
    Relative precision.
  :type eps: float
  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matveca:
    Function to apply the matrix adjoint to a vector, with call signature
    `y = matveca(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matveca: function

  :return:
    Rank of ID.
  :rtype: int
  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  proj = np.empty(m + 1 + 2*n*(min(m,n) + 1), dtype='complex128')
  k, idx, proj, ier = _id.idzp_rid(eps, m, n, matveca, proj)
  if ier != 0: raise RETCODE_ERROR
  proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
  return k, idx, proj

def idz_findrank(eps, m, n, matveca):
  """
  Estimate rank of a complex matrix to a specified relative precision using
  random matrix-vector multiplication.

  :param eps:
    Relative precision.
  :type eps: float
  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matveca:
    Function to apply the matrix adjoint to a vector, with call signature
    `y = matveca(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matveca: function

  :return:
    Rank estimate.
  :rtype: int
  """
  k, ra, ier = _id.idz_findrank(eps, m, n, matveca)
  if ier != 0: raise RETCODE_ERROR
  return k

#-------------------------------------------------------------------------------
# idzp_rsvd.f
#-------------------------------------------------------------------------------

def idzp_rsvd(eps, m, n, matveca, matvec):
  """
  Compute SVD of a complex matrix to a specified relative precision using random
  matrix-vector multiplication.

  :param eps:
    Relative precision.
  :type eps: float
  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matveca:
    Function to apply the matrix adjoint to a vector, with call signature
    `y = matveca(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matveca: function
  :param matvec:
    Function to apply the matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  k, iU, iV, iS, w, ier = _id.idzp_rsvd(eps, m, n, matveca, matvec)
  if ier != 0: raise RETCODE_ERROR
  U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
  V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
  S = w[iS-1:iS+k-1]
  return U, V, S

#-------------------------------------------------------------------------------
# idzr_aid.f
#-------------------------------------------------------------------------------

def idzr_aid(A, k):
  """
  Compute ID of a complex matrix to a specified rank using random sampling.

  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of ID.
  :type k: int

  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=False, dtype='complex128', order='F')
  m, n = A.shape
  w = idzr_aidi(m, n, k)
  idx, proj = _id.idzr_aid(A, k, w)
  if k == n: proj = np.array([], dtype='complex128')
  proj = proj.reshape((k, n-k), order='F')
  return idx, proj

def idzr_aidi(m, n, k):
  """
  Initialize array for :func:`idzr_aid`.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param k:
    Rank of ID.
  :type k: int

  :return:
    Initialization array to be used by :func:`idzr_aid`.
  :rtype: :class:`numpy.ndarray`
  """
  return _id.idzr_aidi(m, n, k)

#-------------------------------------------------------------------------------
# idzr_asvd.f
#-------------------------------------------------------------------------------

def idzr_asvd(A, k):
  """
  Compute SVD of a complex matrix to a specified rank using random sampling.

  :param A:
    Matrix.
  :type A: :class:`numpy.ndarray`
  :param k:
    Rank of SVD.
  :type k: int

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  A = np.array(A, copy=False, dtype='complex128', order='F')
  m, n = A.shape
  w = np.empty((2*k + 22)*m + (6*k + 21)*n + 8*k**2 + 10*k + 90,
               dtype='complex128')
  w_ = idzr_aidi(m, n, k)
  w[:w_.size] = w_
  U, V, S, ier = _id.idzr_asvd(A, k, w)
  if ier != 0: raise RETCODE_ERROR
  return U, V, S

#-------------------------------------------------------------------------------
# idzr_rid.f
#-------------------------------------------------------------------------------

def idzr_rid(m, n, matveca, k):
  """
  Compute ID of a complex matrix to a specified rank using random matrix-vector
  multiplication.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matveca:
    Function to apply the matrix adjoint to a vector, with call signature
    `y = matveca(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matveca: function
  :param k:
    Rank of ID.
  :type k: int

  :return:
    Column index array.
  :rtype: :class:`numpy.ndarray`
  :return:
    Interpolation coefficients.
  :rtype: :class:`numpy.ndarray`
  """
  idx, proj = _id.idzr_rid(m, n, matveca, k)
  proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
  return idx, proj

#-------------------------------------------------------------------------------
# idzr_rsvd.f
#-------------------------------------------------------------------------------

def idzr_rsvd(m, n, matveca, matvec, k):
  """
  Compute SVD of a complex matrix to a specified rank using random matrix-vector
  multiplication.

  :param m:
    Matrix row dimension.
  :type m: int
  :param n:
    Matrix column dimension.
  :type n: int
  :param matveca:
    Function to apply the matrix adjoint to a vector, with call signature
    `y = matveca(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matveca: function
  :param matvec:
    Function to apply the matrix to a vector, with call signature
    `y = matvec(x)`, where `x` and `y` are the input and output vectors,
    respectively.
  :type matvec: function
  :param k:
    Rank of SVD.
  :type k: int

  :return:
    Left singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Right singular vectors.
  :rtype: :class:`numpy.ndarray`
  :return:
    Singular values.
  :rtype: :class:`numpy.ndarray`
  """
  U, V, S, ier = _id.idzr_rsvd(m, n, matveca, matvec, k)
  if ier != 0: raise RETCODE_ERROR
  return U, V, S