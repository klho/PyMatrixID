Tutorial
========

We now present a tutorial on using PyMatrixID. From here on, we will therefore assume that PyMatrixID has been properly installed; if this is not the case, please go back to :doc:`install`.

Overview
--------

The Python interface is located in the directory ``python``, which contains the directory ``pymatrixid``, organizing the main Python package, and ``id_dist.so``, the F2PY-ed Fortran library. The file ``id_dist.so`` contains all wrapped routines and is imported by :mod:`pymatrixid.backend`, which in turn is imported by :mod:`pymatrixid` to create a more convenient interface around it. For details on the Python modules, please see the :doc:`api`.

We will now step through the process of using PyMatrixID, following the driver program as a guide.

Initializing
------------

The first step is to import :mod:`pymatrixid` by issuing the command::

>>> import pymatrixid

at the Python prompt. This should work if you are in the ``python`` directory; otherwise, you may have to first type something like:

>>> import sys
>>> sys.path.append(/path/to/pymatrixid/python/)

in order to tell Python where to look.

Now let's build a matrix. For this, we consider a Hilbert matrix, which is well know to have low rank::

  >>> import numpy as np
  >>> m = n = 1000
  >>> A = np.empty((m, n), order='F')
  >>> for j in range(n):
  ...     for i in range(m):
  ...         A[i,j] = 1. / (i + j + 1)

Note the use of the flag ``order='F'`` in :func:`numpy.array`. This instantiates the matrix in Fortran-contiguous order and is important for avoiding data copying when passing to the backend.

We also define multiplication routines for the matrix::

>>> def matvec (x): return np.dot(A,   x)
>>> def matveca(x): return np.dot(A.T, x)

describing its action and that of its adjoint, respectively, on a vector.

Computing an ID
---------------

We have several choices of algorithm to compute an ID. These fall largely according to two dichotomies:

1. how the matrix is represented, i.e., via its entries or via its action on a vector; and
2. whether to approximate it to a fixed relative precision or to a fixed rank.

We step through each choice in turn below.

From matrix entries
...................

First, we consider a matrix given in terms of its entries.

To compute an ID to a fixed precision, type::

>>> k, idx, proj = pymatrixid.id(A, eps)

where ``eps < 1`` is the desired precision.

To compute an ID to a fixed rank, use::

>>> idx, proj = pymatrixid.id(A, k)

where ``k >= 1`` is the desired rank.

Both algorithms use random sampling and are usually faster than the corresponding older, deterministic algorithms, which can be accessed via the commands::

>>> k, idx, proj = pymatrixid.id(A, eps, rand=False)

and::

>>> idx, proj = pymatrixid.id(A, k, rand=False)

respectively.

From matrix action
..................

Now consider a matrix given in terms of its action on a vector.

To compute an ID to a fixed precision, type::

>>> k, idx, proj = pymatrixid.id(m, n, matveca, eps)

To compute an ID to a fixed rank, use::

>>> idx, proj = pymatrixid.id(m, n, matveca, k)

Reconstructing an ID
--------------------

The ID routines above do not output the skeleton and interpolation matrices explicitly but instead return the relevant information in a more compact (and sometimes more useful) form. To build these matrices, write::

>>> B = pymatrixid.reconskel(A, k, idx)

for the skeleton matrix and::

>>> P = pymatrixid.reconint(idx, proj)

for the interpolation matrix. The ID approximation can then be computed as::

>>> C = np.dot(B, P)

This can also be constructed directly using::

>>> C = pymatrixid.reconid(B, idx, proj)

without having to first compute :math:`P`.

It is also possible to reconstruct the ID explicitly using::

  B = A[:,idx[:k]-1]
  P = np.hstack([np.eye(k), proj])
  C = np.dot(B, P)[:,np.argsort(idx)]

Computing an SVD
----------------

An ID can be converted to an SVD via the command::

>>> U, S, V = pymatrixid.id2svd(B, idx, proj)

The SVD approximation is then::

>>> C = np.dot(U, np.dot(np.diag(S), np.dot(V.T)))

Alternatively, the SVD can be computed "fresh" by combining both the ID and conversion steps into one command. Following the various ID algorithms above, there are correspondingly various SVD algorithms that one can employ.

From matrix entries
...................

We consider first SVD algorithms for a matrix given in terms of its entries.

To compute an SVD to a fixed precision, type::

>>> U, S, V = pymatrixid.svd(A, eps)

To compute an SVD to a fixed rank, use::

>>> U, S, V = pymatrixid.svd(A, k)

Both algorithms use random sampling; for the determinstic versions, issue the keyword ``rand=False`` as above.

From matrix action
..................

Now consider a matrix given in terms of its action on a vector.

To compute an SVD to a fixed precision, type::

>>> U, S, V = pymatrixid.svd(m, n, matvec, matveca, eps)

To compute an SVD to a fixed rank, use::

>>> U, S, V = pymatrixid.svd(m, n, matvec, matveca, k)

Utility routines
----------------

Several utility routines are also available.

To estimate the spectral norm of a matrix, use::

>>> snorm = pymatrixid.snorm(m, n, matvec, matveca)

This algorithm is based on the randomized power method and thus requires only matrix-vector products. The number of iterations to take can be set using the keyword ``its`` (default: ``its=20``).

The same algorithm can also estimate the spectral norm of the difference of two matrices :math:`A_{1}` and :math:`A_{2}` as follows:

>>> diff = pymatrixid.diffsnorm(m, n, matvec1, matvec2, matveca1, matveca2)

where the functions ``matvec1``, ``matvec2``, ``matveca1``, and ``matveca2`` apply the matrices :math:`A_{1}` or :math:`A_{2}` or their adjoints as appropriate. This is often useful for checking the accuracy of a matrix approximation.

Some routines in :mod:`pymatrixid` require estimating the rank of a matrix as well. This can be done with either::

>>> k = pymatrixid.estrank(A, eps)

or::

>>> k = pymatrixid.estrank(m, n, matveca, eps)

depending on the representation.

Finally, the random number generation required for all randomized routines can be controlled via :func:`pymatrixid.rand`. To reset the seed values to their original values, use::

>>> pymatrixid.rand()

To specify the seed values, use::

>>> pymatrixid.rand(s)

where ``s`` must be an array of 55 floats. To simply generate some random numbers, type::

>>> pymatrixid.rand(n)

where ``n`` is the number of random numbers to generate.

Remarks
-------

The above functions all automatically detect the appropriate interface and work with both real and complex data types, passing input arguments to the proper backend routine.

All backend functions can be accessed via the :mod:`pymatrixid.backend` module, which wraps the Fortran functions directly, perhaps with some minor simplification.