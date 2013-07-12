LIB       = pymatrixid
ID_LIB    = id_dist
F2PY      = f2py
F2PYFLAGS = --fcompiler=gnu95 --link-lapack_opt
PYTHON    = python

SRC      = src
BIN      = bin
PYTHON   = python
DOC      = doc
EXAMPLES = examples
ID_DIR   = external/$(ID_LIB)
ID_SRC   = $(ID_DIR)/src

vpath %.pyf $(SRC)
vpath %.so  $(PYTHON)

.PHONY: all python id_dist doc clean clean_python clean_id_dist clean_doc rebuild help

all: python

$(ID_LIB).so: $(ID_LIB).pyf
	$(F2PY) -c $< $(F2PYFLAGS) $(ID_SRC)/*.f
	mv $(ID_LIB).so $(BIN)
	cd $(PYTHON) ; ln -fs ../$(BIN)/$(ID_LIB).so

python: $(ID_LIB).so

id_dist:
	cd $(ID_DIR) ; make

doc: python
	cd $(DOC) ; make html ; make latexpdf

driver: python
	cd $(EXAMPLES) ; python driver.py

clean: clean_python clean_id_dist clean_doc clean_driver

clean_python:
	cd $(BIN) ; rm -f $(ID_LIB).so
	cd $(PYTHON) ; rm -f $(ID_LIB).so
	cd $(PYTHON)/$(LIB) ; rm -f __init__.pyc backend.pyc

clean_id_dist:
	cd $(ID_DIR) ; make clean

clean_doc:
	cd $(DOC) ; make clean

clean_driver:
	cd $(DOC) ; rm -f driver.pyc

rebuild: clean all

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all          to make the Python wrapper"
	@echo "  python       to make the Python wrapper"
	@echo "  id_dist      to make the Fortran ID library"
	@echo "  doc          to make HTML and PDF documentation"
	@echo "  driver       to make the Python driver program"
	@echo "  clean        to remove all compiled objects"
	@echo "  clean_python to remove all compiled Python objects"
	@echo "  clean_doc    to remove all compiled documentation"
	@echo "  clean_driver to remove all compiled driver executables"
	@echo "  rebuild      to clean and rebuild all libraries"