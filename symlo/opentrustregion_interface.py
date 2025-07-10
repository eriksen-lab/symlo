#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
OpenTrustRegion Interface
"""

from __future__ import annotations

__author__ = "Jonas Greiner, Technical University of Denmark, Denmark"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Jonas Greiner"
__email__ = "jongr@dtu.dk"
__status__ = "Development"

import numpy as np
from pyscf.soscf import ciah
from pyopentrustregion import solver_py_interface
from typing import TYPE_CHECKING

from symlo.symmetrization import SymCls, SymCls_all, SymCls_eqv

if TYPE_CHECKING:
    from typing import Tuple, Optional, Callable


class SymCls_OTR(SymCls):
    norb: int
    mo_coeff: np.ndarray

    # unpack matrix
    def unpack(self, kappa: np.ndarray) -> np.ndarray:
        matrix = np.zeros(2 * (self.norb,), dtype=np.float64)
        idx = np.tril_indices(self.norb, -1)
        matrix[idx] = kappa
        return matrix - matrix.conj().T

    # cost function
    def func(self, kappa: np.ndarray) -> float:
        u = ciah.expmat(self.unpack(kappa))
        return self.cost_function(u)[0]

    # cost function, gradient, Hessian diagonal and Hessian linear transformation
    # function
    def update_orbs(
        self, kappa: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        u = ciah.expmat(self.unpack(kappa))
        func, self.g_max = self.cost_function(u)
        grad, hess_x, hdiag = self.gen_g_hop(u)
        self.mo_coeff = self.mo_coeff @ u
        return func, 2 * grad, 2 * hdiag, lambda x: 2 * hess_x(x)

    # kernel function
    def kernel(
        self, callback: Optional[Callable] = None, verbose: Optional[int] = None
    ) -> Tuple[np.ndarray, bool, float]:
        from pyscf.tools import mo_mapping

        # ensure MO coefficients are provided and orbitals can be optimized
        assert self.mo_coeff is not None
        if self.mo_coeff.shape[1] <= 1:
            return self.mo_coeff, True, 0.0
        
        # save starting mo coefficients
        mo_coeff_start = self.mo_coeff.copy()

        # number of orbitals
        self.norb = self.mo_coeff.shape[1]

        # number of parameters
        self.n_param = (self.norb - 1) * self.norb // 2

        # call solver
        solver_py_interface(
            self.func,
            self.update_orbs,
            self.n_param,
            getattr(self, "precond", None),
            getattr(self, "conv_check", None),
            getattr(self, "stability", None),
            getattr(self, "line_search", None),
            getattr(self, "davidson", None),
            getattr(self, "jacobi_davidson", None),
            getattr(self, "prefer_jacobi_davidson", None),
            getattr(self, "conv_tol", None),
            getattr(self, "n_random_trial_vectors", None),
            getattr(self, "start_trust_radius", None),
            getattr(self, "n_macro", None),
            getattr(self, "n_micro", None),
            getattr(self, "global_red_factor", None),
            getattr(self, "local_red_factor", None),
            getattr(self, "seed", None),
            getattr(self, "verbose", None),
            getattr(self, "logger", None),
        )
        finished = True

        # Sort the symmetrized orbitals to make each orbital as close as
        # possible to the corresponding input orbitals
        u0 = mo_coeff_start.T @ self.mol.intor('int1e_ovlp') @ self.mo_coeff
        sorted_idx = mo_mapping.mo_1to1map(u0)
        self.mo_coeff = self.mo_coeff[:, sorted_idx]

        return self.mo_coeff, finished, self.g_max
    

class SymCls_all_OTR(SymCls_all, SymCls_OTR):
    def kernel(
        self, callback: Optional[Callable] = None, verbose: Optional[int] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        this function calls the parent class kernel function and checks whether the
        algorithm has converged
        """
        mo_coeff, finished, g_max = SymCls_OTR.kernel(self, callback, verbose)

        if g_max >= self.conv_tol:
            self.log.info("Restarting symmetrization")
            mo_coeff, finished, g_max = SymCls_OTR.kernel(self, callback, verbose)

        if not finished:
            self.log.warn(
                "Symmetrization of symmetry-equivalent orbitals within "
                "symmetry-invariant blocks has not converged. Try increasing max_cycle "
                "or reducing symm_eqv_thresh."
            )
            raise RuntimeError

        return mo_coeff, finished, g_max


class SymCls_eqv_OTR(SymCls_eqv, SymCls_OTR):
    def kernel(
        self, callback: Optional[Callable] = None, verbose: Optional[int] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        this function calls the parent class kernel function and checks whether the
        algorithm has converged
        """
        mo_coeff, finished, g_max = SymCls_OTR.kernel(self, callback, verbose)

        if g_max >= self.conv_tol:
            self.log.info("Restarting symmetrization")
            mo_coeff, finished, g_max = SymCls_OTR.kernel(self, callback, verbose)

        if not finished:
            self.log.error(
                "Symmetrization of symmetry-invariant blocks has not converged. Try "
                "increasing max_cycle or reducing inv_block_thresh."
            )
            raise RuntimeError

        return mo_coeff, finished, g_max
