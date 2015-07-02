Installing
==========

This section describes how to compile and install PyMatrixID on Unix-like systems. Primary prerequisites include `Git <http://git-scm.com/>`_, `GNU Make <http://www.gnu.org/software/make/>`_, a Fortran compiler such as `GFortran <http://gcc.gnu.org/wiki/GFortran>`_, `F2PY <http://www.scipy.org/F2py>`_, `Python <http://www.python.org/>`_, and `NumPy <http://www.numpy.org/>`_. Secondary prerequisites include `Sphinx <http://sphinx-doc.org/>`_ and `LaTeX <http://www.latex-project.org/>`_ for the documentation.

PyMatrixID has only been tested using GFortran; the use of all other compilers should be considered "at your own risk" (though they should really be fine).

Code repository
---------------

All source files for PyMatrixID (including those for this documentation) are available at https://github.com/klho/PyMatrixID. To download PyMatrixID using Git, type the following command at the shell prompt::

$ git clone https://github.com/klho/PyMatrixID /path/to/local/repository/

Compiling
---------

There are several targets available to compile, namely:

- the Python wrapper;

- the ID package; and

- this documentation.

To see all available targets, switch the working directory to the root of the local repository and type::

$ make help

Hopefully the instructions are self-explanatory; for more explicit directions, please see below. Before beginning, view and edit the file ``Makefile`` to ensure that all options are properly set for your system. In particular, if you will not be using GFortran, be sure to set an alternate compiler as appropriate.

To compile the Python wrapper, type::

$ make

or::

$ make all

or::

$ make python

This creates the F2PY-ed library ``bin/id_dist.so``.

To compile the ID package, type::

$ make id_dist

It is not necessary to compile the ID package in order to use the Python wrapper; the required binaries are created automatically by F2PY. However, compiling the ID package itself may be useful, for example, to test the library independently. The ID package is located in the directory ``external/id_dist``.

To compile the documentation files, type::

$ make doc

Output HTML and PDF files are placed in the directory ``doc``.

Driver program
--------------

PyMatrixID also contains a driver program to demonstrate its use. To run the driver, type::

$ make driver

The driver program is discussed in more detail in :doc:`tutorial`.