PyMatrixID
==========

An interpolative decomposition (ID) of a matrix is a factorization as a product of a skeleton matrix consisting of a subset of columns and an interpolation matrix containing the identity. Like the singular value decomposition (SVD), the ID is a powerful approximation tool. The principal advantages of using an ID instead of an SVD are that:

- it is cheaper to construct;
- it preserves the matrix structure; and
- it is more efficient to compute with in light of the structure of the interpolation matrix.

The ID software package by Martinsson, Rokhlin, Shkolnisky, and Tygert is a Fortran library to compute IDs using various algorithms, including the deterministic rank-revealing QR approach and more recent randomized methods. PyMatrixID is a Python wrapper for this package that exposes its functionality in a more convenient manner. Note that PyMatrixID does not add any functionality beyond that of organizing a simpler and more consistent interface.

PyMatrixID is freely available under the BSD license; for alternate licenses, please contact the author.