SymLo: Code for the Symmetrization of Localized Molecular Orbitals
============================================

Authors
-------

* Jonas Greiner (Johannes Gutenberg University Mainz, author)
* Dr. Janus Juul Eriksen (Technical University of Denmark, author)

Prerequisites
-------------

* [Python](https://www.python.org/) 3.7 or higher
* [PySCF](https://pyscf.org/) 1.6 or higher and its requirements

Usage
-----

SymLo expects both itself and PySCF to be properly exported to Python. This can either
be achieved by installing through pip (in the case of PySCF), having a 
symlo.pth/pyscf.pth file with the corresponding path in the lib/Python3.X/site-packages 
directory of your Python distribution or by including the paths in the environment 
variable `$PYTHONPATH`.\
Once these requirements are satisfied, SymLo can be started by importing and calling 
the symmetrize_mos function and passing input data as arguments and keyword arguments. 
The function will require the following arguments:

* mol: [pyscf](https://pyscf.org/) gto.Mole object
* mo_coeff: mo coefficients, np.ndarray
* point_group: point group, str

The function will additionally accept the following optional keyword arguments:

* verbose: controls the verbosity of the code, should be at least 4 for normal printing, 0 < int < 10, default: mol.verbose
* max_cycle: parameter controls the maximum number of cycles for the symmetrization, int > 0, default: 100
* conv_tol: convergence criterion for the symmetrization, float > 0.0, default: 1.e-13
* inv_block_thresh: threshold for the symmetrization of symmetry-invariant blocks, higher numbers mean that the algorithm is more strict in the orbitals it considers to be symmetric, 0.0 < float < 1.0, default: 0.3
* symm_eqv_thresh: threshold for the symmetrization of symmetry-equivalent orbitals within symmetry-invariant blocks, higher numbers mean that the algorithm is more strict in the orbitals it considers to be symmetric, 0.0 < float < 1.0, default: 0.3
* heatmap: parameter that controls the generation of numpy files for the generation of heatmaps to visualize the symmetrization, bool, default: False

After successful completion, the symmetrize_mos function will return a tuple of two 
variables: 

* symm_eqv_mos: list of lists that describes tuples of symmetry-equivalent orbitals for every symmetry operation, every tuple has two tuples of orbitals as elements that are transformed into each other using the respective symmetry operation
* symm_mo_coeff: mo coefficients after symmetrization, np.ndarray

Tutorials
---------

None at the moment, but please have a look at the [examples](symlo/examples/) that 
accompany the code.

Citing PyMBE
------------

The following paper documents the development of the SymLo code and the theory 
it implements:

* Symmetrization of Localized Molecular Orbitals\
Greiner, J., Eriksen, J. J.\
[J. Phys. Chem. A, (2023)](https://pubs.acs.org/doi/full/10.1021/acs.jpca.3c01371) ([arXiv:2302.13654](https://arxiv.org/abs/2302.13654))

Bug reports
-----------

Bugs can be reported by submitting an [issue](https://github.com/jonas-greiner/symlo/issues)
