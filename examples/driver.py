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

  # find rank
  S = np.linalg.svd(A, compute_uv=False)
  try:    rank = np.nonzero(S < eps)[0][0]
  except: rank = n

  # print input summary
  print("Hilbert matrix dimension:        {:8d}".format(n))
  print("Working precision:               {:8.2e}".format(eps))
  print("Rank to working precision:       {:8d}".format(rank))

  # convenience function to summarize each sub-test
  def summarize(t, A, B):
    print("{:8.2e} (s) / {:>5}".format(t, str(np.allclose(A, B, eps))))

  # convenience function to perform tests for a given type
  def test(desc, dz, A):
    desc = desc.capitalize()
    L = aslinearoperator(A)

    # test ID routines
    print("-----------------------------------------")
    print("{} ID routines".format(desc))
    print("-----------------------------------------")

    # fixed precision
    print("Calling id{}p_id  ...".format(dz), end=" ")
    t0 = time.perf_counter()
    k, idx, proj = pymatrixid.interp_decomp(A, eps, rand=False)
    t = time.perf_counter() - t0
    B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
    summarize(t, A, B)

    print("Calling id{}p_aid ...".format(dz), end=" ")
    t0 = time.perf_counter()
    k, idx, proj = pymatrixid.interp_decomp(A, eps)
    t = time.perf_counter() - t0
    B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
    summarize(t, A, B)

    print("Calling id{}p_rid ...".format(dz), end=" ")
    t0 = time.perf_counter()
    k, idx, proj = pymatrixid.interp_decomp(L, eps)
    t = time.perf_counter() - t0
    B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
    summarize(t, A, B)

    # fixed rank
    k = rank

    print("Calling id{}r_id  ...".format(dz), end=" ")
    t0 = time.perf_counter()
    idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
    t = time.perf_counter() - t0
    B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
    summarize(t, A, B)

    print("Calling id{}r_aid ...".format(dz), end=" ")
    t0 = time.perf_counter()
    idx, proj = pymatrixid.interp_decomp(A, k)
    t = time.perf_counter() - t0
    B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
    summarize(t, A, B)

    print("Calling id{}r_rid ...".format(dz), end=" ")
    t0 = time.perf_counter()
    idx, proj = pymatrixid.interp_decomp(L, k)
    t = time.perf_counter() - t0
    B = pymatrixid.reconstruct_matrix_from_id(A[:,idx[:k]], idx, proj)
    summarize(t, A, B)

    # test SVD routines
    print("-----------------------------------------")
    print("{} SVD routines".format(desc))
    print("-----------------------------------------")

    # fixed precision
    print("Calling id{}p_svd ...".format(dz), end=" ")
    t0 = time.perf_counter()
    U, S, V = pymatrixid.svd(A, eps, rand=False)
    t = time.perf_counter() - t0
    B = U @ np.diag(S) @ V.conj().T
    summarize(t, A, B)

    print("Calling id{}p_asvd...".format(dz), end=" ")
    t0 = time.perf_counter()
    U, S, V = pymatrixid.svd(A, eps)
    t = time.perf_counter() - t0
    B = U @ np.diag(S) @ V.conj().T
    summarize(t, A, B)

    print("Calling id{}p_rsvd...".format(dz), end=" ")
    t0 = time.perf_counter()
    U, S, V = pymatrixid.svd(L, eps)
    t = time.perf_counter() - t0
    B = U @ np.diag(S) @ V.conj().T
    summarize(t, A, B)

    # fixed rank
    k = rank

    print("Calling id{}r_svd ...".format(dz), end=" ")
    t0 = time.perf_counter()
    U, S, V = pymatrixid.svd(A, k, rand=False)
    t = time.perf_counter() - t0
    B = U @ np.diag(S) @ V.conj().T
    summarize(t, A, B)

    print("Calling id{}r_asvd...".format(dz), end=" ")
    t0 = time.perf_counter()
    U, S, V = pymatrixid.svd(A, k)
    t = time.perf_counter() - t0
    B = U @ np.diag(S) @ V.conj().T
    summarize(t, A, B)

    print("Calling id{}r_rsvd...".format(dz), end=" ")
    t0 = time.perf_counter()
    U, S, V = pymatrixid.svd(L, k)
    t = time.perf_counter() - t0
    B = U @ np.diag(S) @ V.conj().T
    summarize(t, A, B)

  # test real routines
  test("real", "d", A)

  # complexify Hilbert matrix
  A = A*(1 + 1j)

  # test complex routines
  test("complex", "z", A)
