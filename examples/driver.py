#*******************************************************************************
#   Copyright (C) 2013 Kenneth L. Ho
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

import sys
sys.path.append('../python/')

import pymatrixid
import numpy as np
from scipy.linalg import hilbert
from scipy.sparse.linalg import aslinearoperator
import time

if __name__ == '__main__':
  """
  Test ID routines on a Hilbert matrix.
  """
  # set parameters
  n = 1000
  eps = 1e-12

  # construct Hilbert matrix
  A = hilbert(n)
  L = aslinearoperator(A)

  # find rank
  S = np.linalg.svd(A, compute_uv=False)
  try:    rank = np.nonzero(S < eps)[0][0]
  except: rank = n

  # print input summary
  print "Hilbert matrix dimension:        %8i" % n
  print "Working precision:               %8.2e" % eps
  print "Rank to working precision:       %8i" % rank

  # set print format
  fmt = "%8.2e (s) / %5s"

  # test real ID routines
  print "-----------------------------------------"
  print "Real ID routines"
  print "-----------------------------------------"

  # fixed precision
  print "Calling iddp_id  ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.interp_decomp(A, eps, rand=False)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling iddp_aid ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.interp_decomp(A, eps)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling iddp_rid ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.interp_decomp(L, eps)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  # fixed rank
  k = rank

  print "Calling iddr_id  ...",
  t0 = time.clock()
  idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling iddr_aid ...",
  t0 = time.clock()
  idx, proj = pymatrixid.interp_decomp(A, k)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling iddr_rid ...",
  t0 = time.clock()
  idx, proj = pymatrixid.interp_decomp(L, k)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  # test real SVD routines
  print "-----------------------------------------"
  print "Real SVD routines"
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
  U, S, V = pymatrixid.svd(L, eps)
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
  U, S, V = pymatrixid.svd(L, k)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.T))
  print fmt % (t, np.allclose(A, B, eps))

  # complexify Hilbert matrix
  A = A*(1 + 1j)
  L = aslinearoperator(A)

  # test complex ID routines
  print "-----------------------------------------"
  print "Complex ID routines"
  print "-----------------------------------------"

  # fixed precision
  print "Calling idzp_id  ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.interp_decomp(A, eps, rand=False)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling idzp_aid ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.interp_decomp(A, eps)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling idzp_rid ...",
  t0 = time.clock()
  k, idx, proj = pymatrixid.interp_decomp(L, eps)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  # fixed rank
  k = rank

  print "Calling idzr_id  ...",
  t0 = time.clock()
  idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling idzr_aid ...",
  t0 = time.clock()
  idx, proj = pymatrixid.interp_decomp(A, k)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling idzr_rid ...",
  t0 = time.clock()
  idx, proj = pymatrixid.interp_decomp(L, k)
  t = time.clock() - t0
  B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
  print fmt % (t, np.allclose(A, B, eps))

  # test complex SVD routines
  print "-----------------------------------------"
  print "Complex SVD routines"
  print "-----------------------------------------"

  # fixed precision
  print "Calling idzp_svd ...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(A, eps, rand=False)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.conj().T))
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling idzp_asvd...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(A, eps)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.conj().T))
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling idzp_rsvd...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(L, eps)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.conj().T))
  print fmt % (t, np.allclose(A, B, eps))

  # fixed rank
  k = rank

  print "Calling idzr_svd ...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(A, k, rand=False)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.conj().T))
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling idzr_asvd...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(A, k)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.conj().T))
  print fmt % (t, np.allclose(A, B, eps))

  print "Calling idzr_rsvd...",
  t0 = time.clock()
  U, S, V = pymatrixid.svd(L, k)
  t = time.clock() - t0
  B = np.dot(U, np.dot(np.diag(S), V.conj().T))
  print fmt % (t, np.allclose(A, B, eps))