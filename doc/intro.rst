Introduction
============

An interpolative decomposition (ID) of a matrix :math:`A \in \mathbb{C}^{m \times n}` of rank :math:`k \leq \min \{ m, n \}` is a factorization

.. math::
  A \Pi =
  \begin{bmatrix}
   A \Pi_{1} & A \Pi_{2}
  \end{bmatrix} =
  A \Pi_{1}
  \begin{bmatrix}
   I & T
  \end{bmatrix},

where :math:`\Pi = [\Pi_{1}, \Pi_{2}]` is a permutation matrix with :math:`\Pi_{1} \in \{ 0, 1 \}^{n \times k}`, i.e., :math:`A \Pi_{2} = A \Pi_{1} T`. This can equivalently be written as :math:`A = BP`, where :math:`B = A \Pi_{1}` and :math:`P = [I, T] \Pi^{\mathsf{T}}` are the *skeleton* and *interpolation matrices*, respectively.

If :math:`A` does not have exact rank :math:`k`, then there exists an approximation in the form of an ID such that :math:`A = BP + E`, where :math:`\| E \| \sim \sigma_{k + 1}` is on the order of the :math:`(k + 1)`-th largest singular value of :math:`A`. Note that :math:`\sigma_{k + 1}` is the best possible error for a rank-:math:`k` approximation and, in fact, is achieved by the singular value decomposition (SVD) :math:`A \approx U S V^{*}`, where :math:`U \in \mathbb{C}^{m \times k}` and :math:`V \in \mathbb{C}^{n \times k}` have orthonormal columns and :math:`S = \mathop{\mathrm{diag}} (\sigma_{i}) \in \mathbb{C}^{k \times k}` is diagonal with nonnegative entries. The principal advantages of using an ID over an SVD are that:

- it is cheaper to construct;
- it preserves the structure of :math:`A`; and
- it is more efficient to compute with in light of the identity submatrix of :math:`P`.

Overview
--------

The ID software package [4]_ by Martinsson, Rokhlin, Shkolnisky, and Tygert is a Fortran library to compute IDs using various algorithms, including the rank-revealing QR approach of [1]_ and the more recent randomized methods described in [2]_, [3]_, and [5]_. PyMatrixID is a Python wrapper for this package that exposes its functionality in a more convenient manner. Note that PyMatrixID does not add any functionality beyond that of organizing a simpler and more consistent interface.

We advise the user to consult also the documentation for the ID package, which is included in full as part of PyMatrixID.

Licensing and availability
--------------------------

PyMatrixID is freely available under the `BSD license <http://opensource.org/licenses/BSD-3-Clause>`_ and can be downloaded at https://github.com/klho/PyMatrixID. To request alternate licenses, please contact the author.

PyMatrixID also distributes the ID software package, which is likewise released under the BSD license.

References
----------

.. [1] H.\  Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. On the compression of low rank matrices. `SIAM J. Sci. Comput.` 26 (4): 1389--1404, 2005. `doi:10.1137/030602678 <http://dx.doi.org/10.1137/030602678>`_.

.. [2] E.\  Liberty, F. Woolfe, P.G. Martinsson, V. Rokhlin, M. Tygert. Randomized algorithms for the low-rank approximation of matrices. `Proc. Natl. Acad. Sci. U.S.A.` 104 (51): 20167--20172, 2007. `doi:10.1073/pnas.0709640104 <http://dx.doi.org/10.1073/pnas.0709640104>`_.

.. [3] P.G. Martinsson, V. Rokhlin, M. Tygert. A randomized algorithm for the decomposition of matrices. `Appl. Comput. Harmon. Anal.` 30 (1): 47--68,  2011. `doi:10.1016/j.acha.2010.02.003 <http://dx.doi.org/10.1016/j.acha.2010.02.003>`_.

.. [4] P.G. Martinsson, V. Rokhlin, Y. Shkolnisky, M. Tygert. ID: a software package for low-rank approximation of matrices via interpolative decompositions, version 0.2. http://cims.nyu.edu/~tygert/id_doc.pdf.

.. [5] F.\  Woolfe, E. Liberty, V. Rokhlin, M. Tygert. A fast randomized algorithm for the approximation of matrices. `Appl. Comput. Harmon. Anal.` 25 (3): 335--366, 2008. `doi:10.1016/j.acha.2007.12.002 <http://dx.doi.org/10.1016/j.acha.2007.12.002>`_.