#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
symmetrization class module
"""

from __future__ import annotations

__author__ = "Jonas Greiner, Technical University of Denmark, Denmark"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Jonas Greiner"
__email__ = "jongr@dtu.dk"
__status__ = "Development"

import numpy as np
from pyscf import gto, lib
from pyscf.soscf import ciah
from pyscf.lib import logger
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Tuple, Dict, List, Optional, Callable, Union


MAX_CONV = 1 / np.sqrt(1e-14)


class SymCls(lib.StreamObject, ciah.CIAHOptimizerMixin):
    r"""
    The symmetrization optimizer that minimizes blocks of the symmetry operation
    transformation matrix

    Args:
        mol : Mole object

    Attributes for SymCls:
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB. Default value equals to :class:`Mole.max_memory`.
        conv_tol : float
            Converge threshold. Default 1e-13
        max_cycle : int
            The max. number of macro iterations. Default 100
        max_iters : int
            The max. number of iterations in each macro iteration. Default 20
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is preferred.
            Default 0.03

    Saved results

        mo_coeff : ndarray
            Symmetrized orbitals

    """

    conv_tol = 1e-13
    max_cycle = 100
    max_iters = 20
    max_stepsize = 0.05
    ah_trust_region = 3
    ah_start_tol = 1e9
    ah_max_cycle = 40
    ah_lindep = 1e-300

    def __init__(
        self,
        mol: gto.Mole,
        symm_ops: np.ndarray,
        symm_eqv_orbs: Union[
            List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]], List[List[int]]
        ],
        mo_coeff: np.ndarray,
    ):
        ciah.CIAHOptimizerMixin.__init__(self)
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.mo_coeff = mo_coeff

        keys = set(
            (
                "conv_tol",
                "conv_tol_grad",
                "max_cycle",
                "max_iters",
                "max_stepsize",
                "ah_trust_region",
                "ah_start_tol",
                "ah_max_cycle",
                "ah_lindep",
            )
        )
        self._keys = set(self.__dict__.keys()).union(keys)
        self.symm_ops = symm_ops
        self.log = logger.new_logger(self, verbose=self.verbose)

    def dump_flags(self, verbose: Optional[int] = None):
        self.log.info("\n")
        self.log.info("******** %s ********", self.__class__)
        self.log.info("conv_tol = %s", self.conv_tol)
        self.log.info("max_cycle = %s", self.max_cycle)
        self.log.info("max_stepsize = %s", self.max_stepsize)
        self.log.info("max_iters = %s", self.max_iters)
        self.log.info("kf_interval = %s", self.kf_interval)
        self.log.info("kf_trust_region = %s", self.kf_trust_region)
        self.log.info("ah_start_tol = %s", self.ah_start_tol)
        self.log.info("ah_start_cycle = %s", self.ah_start_cycle)
        self.log.info("ah_level_shift = %s", self.ah_level_shift)
        self.log.info("ah_conv_tol = %s", self.ah_conv_tol)
        self.log.info("ah_lindep = %s", self.ah_lindep)
        self.log.info("ah_max_cycle = %s", self.ah_max_cycle)
        self.log.info("ah_trust_region = %s", self.ah_trust_region)

    def kernel(
        self, callback: Optional[Callable] = None, verbose: Optional[int] = None
    ) -> Tuple[np.ndarray, bool, float]:
        from pyscf.tools import mo_mapping

        if self.mo_coeff.shape[1] <= 1:
            return self.mo_coeff, True, 0.0

        self.log.verbose = self.verbose

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        cput0 = (logger.process_clock(), logger.perf_counter())

        conv_tol_grad = np.sqrt(self.conv_tol * 0.1)

        u0 = np.eye(self.mo_coeff.shape[1])

        rotaiter = ciah.rotate_orb_cc(self, u0, conv_tol_grad, verbose=self.log)
        u, _, stat = next(rotaiter)
        cput1 = self.log.timer("initializing CIAH", *cput0)

        tot_kf = stat.tot_kf
        tot_hop = stat.tot_hop
        conv = False
        limit_reached = False
        e_last = 0
        for imacro in range(self.max_cycle):
            u0 = np.dot(u0, u)
            e, e_max = self.cost_function(u0)
            e_last, de = e, e - e_last

            self.log.info(
                "macro= %d  f(x)= %.14g  delta_f= %g  max(Gpq)= %g  %d KF %d Hx",
                imacro + 1,
                e,
                de,
                e_max,
                stat.tot_kf + 1,
                stat.tot_hop,
            )
            cput1 = self.log.timer("cycle= %d" % (imacro + 1), *cput1)

            if e_max < self.conv_tol:
                conv = True

            if de == 0.0:
                limit_reached = True

            if callable(callback):
                callback(locals())

            if conv or limit_reached:
                break

            u, _, stat = rotaiter.send(u0)
            tot_kf += stat.tot_kf
            tot_hop += stat.tot_hop

        rotaiter.close()
        self.log.info(
            "macro X = %d  f(x)= %.14g  max(Gpq)= %g  %d intor %d KF %d Hx",
            imacro + 1,
            e,
            e_max,
            (imacro + 1) * 2,
            tot_kf + imacro + 1,
            tot_hop,
        )

        if conv:
            finished = True
        elif limit_reached:
            self.log.warn(
                "Maximum symmetrization within supplied orbital space reached."
            )
            finished = True
        else:
            finished = False

        # Sort the symmetrized orbitals to make each orbital as close as
        # possible to the corresponding input orbitals
        sorted_idx = mo_mapping.mo_1to1map(u0)
        self.mo_coeff = np.dot(self.mo_coeff, u0[:, sorted_idx])

        return self.mo_coeff, finished, e_max


class SymCls_all(SymCls):
    def __init__(
        self,
        mol: gto.Mole,
        symm_ops: np.ndarray,
        symm_eqv_orbs: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
        mo_coeff: np.ndarray,
    ):
        super().__init__(mol, symm_ops, symm_eqv_orbs, mo_coeff)

        self.incl_orbs: List[List[np.ndarray]] = []
        self.excl_orbs: List[List[np.ndarray]] = []

        for op in range(len(self.symm_ops)):
            self.incl_orbs.append([])
            self.excl_orbs.append([])
            for orb_set in symm_eqv_orbs[op]:
                self.incl_orbs[op].append(np.array(orb_set[0]))
                self.excl_orbs[op].append(
                    np.delete(np.arange(mo_coeff.shape[1]), orb_set[1])
                )

    def gen_g_hop(self, u: np.ndarray):
        """
        this function generates the gradient, hessian diagonal and the function that
        calculates the matrix-vector product of the hessian with some vector x
        """
        # get number of mos
        norb = self.mo_coeff.shape[1]

        # get starting mo coefficients
        mo_coeff_s = self.mo_coeff

        # get current mo coefficients
        mo_coeff_u = np.dot(self.mo_coeff, u)

        # generate intermediates
        interm: List[List[Dict[str, np.ndarray]]] = []
        for op, op_mat in enumerate(self.symm_ops):
            uu = mo_coeff_u.T @ op_mat @ mo_coeff_u
            ss = mo_coeff_s.T @ op_mat @ mo_coeff_s
            us = mo_coeff_u.T @ op_mat @ mo_coeff_s
            su = mo_coeff_s.T @ op_mat @ mo_coeff_u

            interm.append([])

            for incl, excl in zip(self.incl_orbs[op], self.excl_orbs[op]):
                interm[-1].append(
                    {
                        "uu_ia": uu[incl.reshape(-1, 1), excl],
                        "su_pa": su[:, excl],
                        "us_ip": us[incl, :],
                        "ss_pq": ss,
                    }
                )

                interm[-1][-1]["su_pa@su_pa.T"] = (
                    interm[-1][-1]["su_pa"] @ interm[-1][-1]["su_pa"].T
                )
                interm[-1][-1]["us_ip.T@us_ip"] = (
                    interm[-1][-1]["us_ip"].T @ interm[-1][-1]["us_ip"]
                )

        # calculate gradient
        g0 = np.zeros((norb, norb), dtype=np.float64)
        for op in range(len(self.symm_ops)):
            for orbset, (incl, excl) in enumerate(
                zip(self.incl_orbs[op], self.excl_orbs[op])
            ):
                uu_ia = interm[op][orbset]["uu_ia"]
                su_pa = interm[op][orbset]["su_pa"]
                us_ip = interm[op][orbset]["us_ip"]

                g0[incl, :] -= uu_ia @ su_pa.T
                g0[excl, :] -= uu_ia.T @ us_ip

        g = self.pack_uniq_var(g0 - g0.T) * 2

        # calculate hessian diagonal
        h_diag0 = np.zeros((norb, norb), dtype=np.float64)
        for op in range(len(self.symm_ops)):
            for orbset, (incl, excl) in enumerate(
                zip(self.incl_orbs[op], self.excl_orbs[op])
            ):
                us_ij = interm[op][orbset]["us_ip"][:, incl]
                su_ab = interm[op][orbset]["su_pa"][excl, :]
                uu_ia = interm[op][orbset]["uu_ia"]
                ss_ai = interm[op][orbset]["ss_pq"][excl.reshape(-1, 1), incl]

                su_pa_su_paT = interm[op][orbset]["su_pa@su_pa.T"]
                us_ipT_us_ip = interm[op][orbset]["us_ip.T@us_ip"]

                h_diag0[incl, :] += np.diag(su_pa_su_paT)[np.newaxis, :]
                h_diag0[excl, :] += np.diag(us_ipT_us_ip)[np.newaxis, :]
                h_diag0[incl.reshape(-1, 1), excl] -= (
                    np.einsum("ii,aa->ia", us_ij, su_ab)
                    + 2 * np.einsum("ia,ai->ia", uu_ia, ss_ai)
                    + np.einsum("aa,ii->ia", su_ab, us_ij)
                )

        h_diag = self.pack_uniq_var(h_diag0 + h_diag0.T) * 2

        def h_op(x: np.ndarray):
            """
            this function calculates the matrix-vector product with some vector x
            """
            # calculate hx
            x = self.unpack_uniq_var(x)
            hx0 = np.zeros_like(x)
            for op in range(len(self.symm_ops)):
                for orbset, (incl, excl) in enumerate(
                    zip(self.incl_orbs[op], self.excl_orbs[op])
                ):
                    x_ip = x[incl, :]
                    x_ap = x[excl, :]

                    su_pa = interm[op][orbset]["su_pa"]
                    us_ip = interm[op][orbset]["us_ip"]
                    uu_ia = interm[op][orbset]["uu_ia"]
                    ss_pq = interm[op][orbset]["ss_pq"]

                    su_pa_su_paT = interm[op][orbset]["su_pa@su_pa.T"]
                    us_ipT_us_ip = interm[op][orbset]["us_ip.T@us_ip"]

                    hx0[incl, :] += (
                        x_ip @ su_pa_su_paT
                        + us_ip @ x_ap.T @ su_pa.T
                        + uu_ia @ x_ap @ ss_pq.T
                    )
                    hx0[excl, :] += (
                        x_ap @ us_ipT_us_ip
                        + su_pa.T @ x_ip.T @ us_ip
                        + uu_ia.T @ x_ip @ ss_pq
                    )

            return self.pack_uniq_var(hx0 - hx0.T) * 4

        return g, h_op, h_diag

    def get_grad(self, u: Optional[np.ndarray] = None):
        """
        this function calculates the gradient
        """
        # get number of mos
        norb = self.mo_coeff.shape[1]

        # get starting mo coefficients
        mo_coeff_s = self.mo_coeff

        # get current mo coefficients
        if u is None:
            u = np.eye(norb)
        mo_coeff_u = np.dot(self.mo_coeff, u)

        # generate intermediates
        uu: List[np.ndarray] = []
        us: List[np.ndarray] = []
        su: List[np.ndarray] = []
        for op_mat in self.symm_ops:
            uu.append(mo_coeff_u.T @ op_mat @ mo_coeff_u)
            us.append(mo_coeff_u.T @ op_mat @ mo_coeff_s)
            su.append(mo_coeff_s.T @ op_mat @ mo_coeff_u)

        # calculate gradient
        g0 = np.zeros((norb, norb), dtype=np.float64)
        for op in range(len(self.symm_ops)):
            for incl, excl in zip(self.incl_orbs[op], self.excl_orbs[op]):
                uu_ia = uu[op][incl.reshape(-1, 1), excl]
                su_pa = su[op][:, excl]
                us_ip = us[op][incl, :]

                g0[incl, :] -= uu_ia @ su_pa.T
                g0[excl, :] -= uu_ia.T @ us_ip

        return self.pack_uniq_var(g0 - g0.T) * 2

    def cost_function(self, u: Optional[np.ndarray] = None):
        """
        this function calculates the value of the cost function
        """
        # get number of mos
        norb = self.mo_coeff.shape[1]

        # get current mo coefficients
        if u is None:
            u = np.eye(norb)
        mo_coeff_u = np.dot(self.mo_coeff, u)

        # calculate cost function
        p = 0.0
        g_max = 0.0
        for op_mat, incl_orbs, excl_orbs in zip(
            self.symm_ops, self.incl_orbs, self.excl_orbs
        ):
            for incl, excl in zip(incl_orbs, excl_orbs):
                g = mo_coeff_u[:, incl].T @ op_mat @ mo_coeff_u[:, excl]
                if g.size > 0:
                    g_max = max(g_max, np.max(np.abs(g)))
                p += np.sum(g**2)

        return p, g_max

    def kernel(
        self, callback: Optional[Callable] = None, verbose: Optional[int] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        this function calls the parent class kernel function and checks whether the
        algorithm has converged
        """
        mo_coeff, finished, g_max = SymCls.kernel(self, callback, verbose)

        if g_max >= self.conv_tol:
            self.log.info("Restarting symmetrization")
            mo_coeff, finished, g_max = SymCls.kernel(self, callback, verbose)

        if not finished or g_max >= 10 * self.conv_tol:
            self.log.warn(
                "Symmetrization of symmetry-equivalent orbitals within "
                "symmetry-invariant blocks has not converged. Try increasing max_cycle "
                "or reducing symm_eqv_thresh."
            )
            raise RuntimeError

        return mo_coeff, finished, g_max


class SymCls_eqv(SymCls):
    def __init__(
        self,
        mol: gto.Mole,
        symm_ops: np.ndarray,
        symm_eqv_orbs: List[List[int]],
        mo_coeff: np.ndarray,
    ):
        super().__init__(mol, symm_ops, symm_eqv_orbs, mo_coeff)

        self.incl_orbs: List[slice] = []
        self.excl_orbs: List[slice] = []

        for orb_set in symm_eqv_orbs:
            if orb_set[0] > 0:
                self.incl_orbs.append(slice(orb_set[0], orb_set[-1] + 1))
                self.excl_orbs.append(slice(0, orb_set[0]))
            if orb_set[-1] < mo_coeff.shape[1] - 1:
                self.incl_orbs.append(slice(orb_set[0], orb_set[-1] + 1))
                self.excl_orbs.append(slice(orb_set[-1] + 1, mo_coeff.shape[1]))

    def gen_g_hop(self, u: np.ndarray):
        """
        this function generates the gradient, hessian diagonal and the function that
        calculates the matrix-vector product of the hessian with some vector x
        """
        # get number of mos
        norb = self.mo_coeff.shape[1]

        # get starting mo coefficients
        mo_coeff_s = self.mo_coeff

        # get current mo coefficients
        mo_coeff_u = np.dot(self.mo_coeff, u)

        # generate intermediates
        interm: List[List[Dict[str, np.ndarray]]] = []

        su_pa_su_paT = [
            np.zeros((norb, norb), dtype=np.float64) for _ in range(len(self.incl_orbs))
        ]
        us_ipT_us_ip = [
            np.zeros((norb, norb), dtype=np.float64) for _ in range(len(self.incl_orbs))
        ]

        for op, op_mat in enumerate(self.symm_ops):
            uu = mo_coeff_u.T @ op_mat @ mo_coeff_u
            ss = mo_coeff_s.T @ op_mat @ mo_coeff_s
            us = mo_coeff_u.T @ op_mat @ mo_coeff_s
            su = mo_coeff_s.T @ op_mat @ mo_coeff_u

            interm.append([])

            for orbset, (incl, excl) in enumerate(zip(self.incl_orbs, self.excl_orbs)):
                interm[-1].append(
                    {
                        "uu_ia": uu[incl, excl],
                        "su_pa": su[:, excl],
                        "us_ip": us[incl, :],
                        "ss_pq": ss,
                    }
                )

                su_pa_su_paT[orbset] += (
                    interm[-1][-1]["su_pa"] @ interm[-1][-1]["su_pa"].T
                )
                us_ipT_us_ip[orbset] += (
                    interm[-1][-1]["us_ip"].T @ interm[-1][-1]["us_ip"]
                )

        # calculate gradient
        g0 = np.zeros((norb, norb), dtype=np.float64)
        for orbset, (incl, excl) in enumerate(zip(self.incl_orbs, self.excl_orbs)):
            for op in range(len(self.symm_ops)):
                uu_ia = interm[op][orbset]["uu_ia"]
                su_pa = interm[op][orbset]["su_pa"]
                us_ip = interm[op][orbset]["us_ip"]

                g0[incl, :] -= uu_ia @ su_pa.T
                g0[excl, :] -= uu_ia.T @ us_ip

        g = self.pack_uniq_var(g0 - g0.T) * 2

        # calculate hessian diagonal
        h_diag0 = np.zeros((norb, norb), dtype=np.float64)
        for orbset, (incl, excl) in enumerate(zip(self.incl_orbs, self.excl_orbs)):
            for op in range(len(self.symm_ops)):
                us_ij = interm[op][orbset]["us_ip"][:, incl]
                su_ab = interm[op][orbset]["su_pa"][excl, :]
                uu_ia = interm[op][orbset]["uu_ia"]
                ss_ai = interm[op][orbset]["ss_pq"][excl, incl]

                h_diag0[incl, excl] -= (
                    np.einsum("ii,aa->ia", us_ij, su_ab)
                    + 2 * np.einsum("ia,ai->ia", uu_ia, ss_ai)
                    + np.einsum("aa,ii->ia", su_ab, us_ij)
                )

            h_diag0[incl, :] += np.diag(su_pa_su_paT[orbset])[np.newaxis, :]
            h_diag0[excl, :] += np.diag(us_ipT_us_ip[orbset])[np.newaxis, :]

        h_diag = self.pack_uniq_var(h_diag0 + h_diag0.T) * 2

        def h_op(x: np.ndarray):
            """
            this function calculates the matrix-vector product with some vector x
            """
            # calculate hx
            x = self.unpack_uniq_var(x)
            hx0 = np.zeros_like(x)
            for orbset, (incl, excl) in enumerate(zip(self.incl_orbs, self.excl_orbs)):
                x_ip = x[incl, :]
                x_ap = x[excl, :]

                for op in range(len(self.symm_ops)):
                    su_pa = interm[op][orbset]["su_pa"]
                    us_ip = interm[op][orbset]["us_ip"]
                    uu_ia = interm[op][orbset]["uu_ia"]
                    ss_pq = interm[op][orbset]["ss_pq"]

                    hx0[incl, :] += us_ip @ x_ap.T @ su_pa.T + uu_ia @ x_ap @ ss_pq.T
                    hx0[excl, :] += su_pa.T @ x_ip.T @ us_ip + uu_ia.T @ x_ip @ ss_pq

                hx0[incl, :] += x_ip @ su_pa_su_paT[orbset]
                hx0[excl, :] += x_ap @ us_ipT_us_ip[orbset]

            return self.pack_uniq_var(hx0 - hx0.T) * 4

        return g, h_op, h_diag

    def get_grad(self, u: Optional[np.ndarray] = None):
        """
        this function calculates the gradient
        """
        # get number of mos
        norb = self.mo_coeff.shape[1]

        # get starting mo coefficients
        mo_coeff_s = self.mo_coeff

        # get current mo coefficients
        if u is None:
            u = np.eye(norb)
        mo_coeff_u = np.dot(self.mo_coeff, u)

        # generate intermediates
        uu: List[np.ndarray] = []
        us: List[np.ndarray] = []
        su: List[np.ndarray] = []
        for op_mat in self.symm_ops:
            uu.append(mo_coeff_u.T @ op_mat @ mo_coeff_u)
            us.append(mo_coeff_u.T @ op_mat @ mo_coeff_s)
            su.append(mo_coeff_s.T @ op_mat @ mo_coeff_u)

        # calculate gradient
        g0 = np.zeros((norb, norb), dtype=np.float64)
        for op in range(len(self.symm_ops)):
            for incl, excl in zip(self.incl_orbs, self.excl_orbs):
                uu_ia = uu[op][incl, excl]
                su_pa = su[op][:, excl]
                us_ip = us[op][incl, :]

                g0[incl, :] -= uu_ia @ su_pa.T
                g0[excl, :] -= uu_ia.T @ us_ip

        return self.pack_uniq_var(g0 - g0.T) * 2

    def cost_function(self, u: Optional[np.ndarray] = None):
        """
        this function calculates the value of the cost function
        """
        # get number of mos
        norb = self.mo_coeff.shape[1]

        # get current mo coefficients
        if u is None:
            u = np.eye(norb)
        mo_coeff_u = np.dot(self.mo_coeff, u)

        # calculate cost function
        p = 0.0
        g_max = 0.0
        for op_mat in self.symm_ops:
            for incl, excl in zip(self.incl_orbs, self.excl_orbs):
                g = mo_coeff_u[:, incl].T @ op_mat @ mo_coeff_u[:, excl]
                if g.size > 0:
                    g_max = max(g_max, np.max(np.abs(g)))
                p += np.sum(g**2)

        return p, g_max

    def kernel(
        self, callback: Optional[Callable] = None, verbose: Optional[int] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        this function calls the parent class kernel function and checks whether the
        algorithm has converged
        """
        mo_coeff, finished, g_max = SymCls.kernel(self, callback, verbose)

        if g_max >= self.conv_tol:
            self.log.info("Restarting symmetrization")
            mo_coeff, finished, g_max = SymCls.kernel(self, callback, verbose)

        if not finished or g_max >= 10 * self.conv_tol:
            self.log.error(
                "Symmetrization of symmetry-invariant blocks has not converged. Try "
                "increasing max_cycle or reducing inv_block_thresh."
            )
            raise RuntimeError

        return mo_coeff, finished, g_max
    

class SymCls_all_PySCF(SymCls_all):
    def gen_g_hop(self, u: np.ndarray):
        """
        this function generates the gradient, hessian diagonal and the function that
        calculates the matrix-vector product of the hessian with some vector x
        """
        g, h_op, h_diag = super().gen_g_hop(u)

        def scaled_h_op(x: np.ndarray):
            """
            this function calculates the matrix-vector product with some vector x
            """
            return MAX_CONV * h_op(x)

        return MAX_CONV * g, scaled_h_op, MAX_CONV * h_diag

    def get_grad(self, u: Optional[np.ndarray] = None):
        """
        this function calculates the gradient
        """
        return MAX_CONV * super().get_grad(u)

    def cost_function(self, u: Optional[np.ndarray] = None):
        """
        this function calculates the value of the cost function
        """
        p, g_max = super().cost_function(u)
        return MAX_CONV * p, g_max


class SymCls_eqv_PySCF(SymCls_eqv):
    def gen_g_hop(self, u: np.ndarray):
        """
        this function generates the gradient, hessian diagonal and the function that
        calculates the matrix-vector product of the hessian with some vector x
        """
        g, h_op, h_diag = super().gen_g_hop(u)

        def scaled_h_op(x: np.ndarray):
            """
            this function calculates the matrix-vector product with some vector x
            """
            return MAX_CONV * h_op(x)

        return MAX_CONV * g, scaled_h_op, MAX_CONV * h_diag

    def get_grad(self, u: Optional[np.ndarray] = None):
        """
        this function calculates the gradient
        """
        return MAX_CONV * super().get_grad(u)

    def cost_function(self, u: Optional[np.ndarray] = None):
        """
        this function calculates the value of the cost function
        """
        p, g_max = super().cost_function(u)
        return MAX_CONV * p, g_max

