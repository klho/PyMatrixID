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

import sys
sys.path.append('../python/')

import pymatrixid
import numpy as np
import time

if __name__ == '__main__':
  """
  Test ID routines on a Hilbert matrix.
  """
  # construct Hilbert matrix
  m = n = 1000
  A = np.empty((m, n), order='F')
  for j in range(n):
    for i in range(m):
      A[i,j] = 1. / (i + j + 1)

  # define multiplication functions
  def matvec (x): return np.dot(A,   x)
  def matveca(x): return np.dot(A.T, x)

  # set relative precision
  eps = 1e-12

  # find true rank
  S = np.linalg.svd(A, compute_uv=False)
  try:    rank = np.nonzero(S < 1e-12)[0][0]
  except: rank = n

  # print input summary
  print "Hilbert matrix dimension:        %8i" % n
  print "Working precision:               %8.2e" % eps
  print "Rank to working precision:       %8i" % rank
  print "-----------------------------------------"

  # set print format
  fmt = "%8.2e (s) / %5s"

  # test ID routines
  print "ID routines"
  print "-----------------------------------------"

  # fixed precision
  print "Calling iddp_id  ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.id(A, eps, rand=False)
  t = time.clock() - t0
  B = pymatrixid.reconskel(A, k, idx)
  C = pymatrixid.reconid(B, idx, proj)
  print fmt % (t, np.allclose(A, C, eps))

  print "Calling iddp_aid ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.id(A, eps)
  t = time.clock() - t0
  B = pymatrixid.reconskel(A, k, idx)
  C = pymatrixid.reconid(B, idx, proj)
  print fmt % (t, np.allclose(A, C, eps))

  print "Calling iddp_rid ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.id(m, n, matveca, eps)
  t = time.clock() - t0
  B = pymatrixid.reconskel(A, k, idx)
  C = pymatrixid.reconid(B, idx, proj)
  print fmt % (t, np.allclose(A, C, eps))

  # fixed rank
  k = rank

  print "Calling iddr_id  ...",
  t0 = time.clock()
  idx, proj = pymatrixid.id(A, k, rand=False)
  t = time.clock() - t0
  B = pymatrixid.reconskel(A, k, idx)
  C = pymatrixid.reconid(B, idx, proj)
  print fmt % (t, np.allclose(A, C, eps))

  print "Calling iddr_aid ...",
  t0 = time.clock()
  idx, proj = pymatrixid.id(A, k)
  t = time.clock() - t0
  B = pymatrixid.reconskel(A, k, idx)
  C = pymatrixid.reconid(B, idx, proj)
  print fmt % (t, np.allclose(A, C, eps))

  print "Calling iddr_rid ...",
  t0 = time.clock()
  idx, proj = pymatrixid.id(m, n, matveca, k)
  t = time.clock() - t0
  B = pymatrixid.reconskel(A, k, idx)
  C = pymatrixid.reconid(B, idx, proj)
  print fmt % (t, np.allclose(A, C, eps))

  # test SVD routines
  print "-----------------------------------------"
  print "SVD routines"
  print "-----------------------------------------"

  # fixed precision
  print "Calling iddp_svd ...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(A, eps, rand=False)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.T))
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling iddp_asvd...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(A, eps)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.T))
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling iddp_rsvd...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(m, n, matvec, matveca, eps)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.T))
  print fmt % (t, np.allclose(A, B, eps))

  # fixed rank
  k = rank

  print "Calling iddr_svd ...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(A, k, rand=False)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.T))
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling iddr_asvd...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(A, k)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.T))
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling iddr_rsvd...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(m, n, matvec, matveca, k)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.T))
  print fmt % (t, np.allclose(A, B, eps))