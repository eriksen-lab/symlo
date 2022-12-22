#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main symlo module
"""

from __future__ import annotations

__author__ = "Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Jonas Greiner"
__email__ = "jonas.greiner@uni-mainz.de"
__status__ = "Development"

import numpy as np
import scipy as sc
from pyscf import gto, symm, lo, soscf
from pyscf.lib.exceptions import PointGroupSymmetryError
from pyscf.lib import logger
from itertools import combinations
from typing import TYPE_CHECKING
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import seaborn as sns

from symlo.tools import get_symm_op_matrices, get_symm_coord

if TYPE_CHECKING:

    from typing import Tuple, Dict, List, Optional, Callable, Union


COORD_TOL = 1.0e-14
MO_SYMM_TOL = 1.0e-2
SYMM_TOL = 1e-12
DIFF = 1 / np.sqrt(1e-14)


def symm_eqv_mo(
    mol: gto.Mole, mo_coeff: np.ndarray, point_group: str, ncore: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns an array of permutations of symmetry equivalent orbitals for each
    symmetry operation
    """
    # convert point group to standard symbol
    point_group = symm.std_symb(point_group)

    # get number of orbitals
    norb = mo_coeff.shape[1]

    # get number of occupied orbitals
    nocc = max(mol.nelec)

    # get number of virtual orbitals
    nvirt = norb - nocc

    # get ao overlap matrix
    sao = gto.intor_cross("int1e_ovlp", mol, mol)

    # orthogonalize mo coefficients
    mo_coeff = lo.orth.vec_lowdin(mo_coeff, sao)

    # get symmetry transformation matrix in orthogonal ao basis
    trafo_ao = symm_trafo_ao(mol, point_group, sao)

    # get total number of symmetry operations
    nop = trafo_ao.shape[0]

    # initialize array of equivalent mos
    symm_eqv_mos = np.empty((nop, mo_coeff.shape[0]), dtype=np.int64)

    # initialize overlap matrices between mos and transformed mos for all symmetry
    # operations (the individual matrices for each symmetry operation are not symmetric
    # because symmetry operations are not necessarily unitary but their sum is because
    # a group includes an inverse element for every symmetry operation)
    all_symm_trafo_occ_ovlp = np.zeros((nocc - ncore, nocc - ncore))
    all_symm_trafo_virt_ovlp = np.zeros((nvirt, nvirt))

    # loop over symmetry operations
    for op in trafo_ao:

        # transform mos
        op_mo_coeff = op @ mo_coeff

        # add overlap for symmetry operation
        all_symm_trafo_occ_ovlp += np.abs(
            mo_coeff[:, ncore:nocc].T @ op_mo_coeff[:, ncore:nocc]
        )
        all_symm_trafo_virt_ovlp += np.abs(mo_coeff[:, nocc:].T @ op_mo_coeff[:, nocc:])

    # normalize overlap matrix
    all_symm_trafo_occ_ovlp /= nop
    all_symm_trafo_virt_ovlp /= nop

    # set threshold for sparse array
    thresh = 0.5 * np.max(all_symm_trafo_occ_ovlp, axis=0)

    # copy array
    thresh_trafo_occ = all_symm_trafo_occ_ovlp.copy()

    # set elements below threshold to zero
    thresh_trafo_occ[thresh_trafo_occ < thresh] = 0.0

    # create sparse array
    sparse_trafo_occ = sc.sparse.csr_array(thresh_trafo_occ)

    # determine optimal ordering of orbitals
    reorder_occ = sc.sparse.csgraph.reverse_cuthill_mckee(sparse_trafo_occ)

    fig_occ, axs_occ = plt.subplots(
        figsize=(12, 4), ncols=3, constrained_layout=True, sharex=True, sharey=True
    )

    sns.heatmap(
        all_symm_trafo_occ_ovlp, linewidth=0.5, ax=axs_occ[0], norm=LogNorm(), cbar=None
    )
    sns.heatmap(
        all_symm_trafo_occ_ovlp[reorder_occ.reshape(-1, 1), reorder_occ],
        linewidth=0.5,
        ax=axs_occ[1],
        norm=LogNorm(),
        cbar=None,
    )

    # reorder array
    thresh_trafo_occ = thresh_trafo_occ[reorder_occ.reshape(-1, 1), reorder_occ]

    # initialize list for mo blocks that are approximately invariant with respect to
    # all symmetry operations and add first block
    tot_symm_occ_blocks: List[List[int]] = [[0]]

    # initialize row counter
    start = 1

    # perform until all orbitals are considered
    while True:

        # loop over mos
        for mo in range(start, nocc - ncore):

            # check if mos overlaps with last block
            if thresh_trafo_occ[mo, tot_symm_occ_blocks[-1]].any():

                # add mo
                tot_symm_occ_blocks[-1].append(mo)

            else:

                # create new block
                tot_symm_occ_blocks.append([mo])

                # start at next mo
                start = mo + 1

                # block is finished
                break

        else:

            # all orbitals finished
            break

    # set threshold for sparse array
    thresh = 0.5 * np.max(all_symm_trafo_virt_ovlp, axis=0)

    # copy array
    thresh_trafo_virt = all_symm_trafo_virt_ovlp.copy()

    # set elements below threshold to zero
    thresh_trafo_virt[thresh_trafo_virt < thresh] = 0.0

    # create sparse array
    sparse_trafo_virt = sc.sparse.csr_array(thresh_trafo_virt)

    # determine optimal ordering of orbitals
    reorder_virt = sc.sparse.csgraph.reverse_cuthill_mckee(sparse_trafo_virt)

    fig_virt, axs_virt = plt.subplots(
        figsize=(12, 4), ncols=3, constrained_layout=True, sharex=True, sharey=True
    )

    sns.heatmap(
        all_symm_trafo_virt_ovlp,
        linewidth=0.5,
        ax=axs_virt[0],
        norm=LogNorm(),
        cbar=None,
    )
    sns.heatmap(
        all_symm_trafo_virt_ovlp[reorder_virt.reshape(-1, 1), reorder_virt],
        linewidth=0.5,
        ax=axs_virt[1],
        norm=LogNorm(),
        cbar=None,
    )

    # reorder array
    thresh_trafo_virt = thresh_trafo_virt[reorder_virt.reshape(-1, 1), reorder_virt]

    # initialize list for mo blocks that are approximately invariant with respect to
    # all symmetry operations and add first block
    tot_symm_virt_blocks: List[List[int]] = [[0]]

    # initialize starting mo
    start = 1

    # perform until all orbitals are considered
    while True:

        # loop over mos
        for mo in range(start, nvirt):

            # check if mos overlaps with last block
            if thresh_trafo_virt[mo, tot_symm_virt_blocks[-1]].any():

                # add mo
                tot_symm_virt_blocks[-1].append(mo)

            else:

                # create new block
                tot_symm_virt_blocks.append([mo])

                # start at next mo
                start = mo + 1

                # block is finished
                break

        else:

            # all orbitals finished
            break

    print(tot_symm_occ_blocks)
    print(tot_symm_virt_blocks)

    # reorder mo coefficients
    mo_coeff[:, ncore:nocc] = mo_coeff[:, ncore + reorder_occ]
    mo_coeff[:, nocc:] = mo_coeff[:, nocc + reorder_virt]

    # symmetrize mos
    mo_coeff = symmetrize_mos(
        mol,
        mo_coeff,
        nop,
        tot_symm_occ_blocks,
        tot_symm_virt_blocks,
        trafo_ao,
        ncore,
        nocc,
    )

    # initialize overlap matrices between mos and transformed mos for all symmetry
    # operations (the individual matrices for each symmetry operation are not symmetric
    # because symmetry operations are not necessarily unitary but their sum is because
    # a group includes an inverse element for every symmetry operation)
    all_symm_trafo_occ_ovlp = np.zeros((nocc - ncore, nocc - ncore))
    all_symm_trafo_virt_ovlp = np.zeros((nvirt, nvirt))

    # loop over symmetry operations
    for op in trafo_ao:

        # transform mos
        op_mo_coeff = op @ mo_coeff

        # add overlap for symmetry operation
        all_symm_trafo_occ_ovlp += np.abs(
            mo_coeff[:, ncore:nocc].T @ op_mo_coeff[:, ncore:nocc]
        )
        all_symm_trafo_virt_ovlp += np.abs(mo_coeff[:, nocc:].T @ op_mo_coeff[:, nocc:])

    # normalize overlap matrix
    all_symm_trafo_occ_ovlp /= nop
    all_symm_trafo_virt_ovlp /= nop

    sns.heatmap(all_symm_trafo_occ_ovlp, linewidth=0.5, ax=axs_occ[2], norm=LogNorm())

    axs_occ[0].set_title("Before sorting")
    axs_occ[1].set_title("After sorting")
    axs_occ[2].set_title("After symmetrization")

    fig_occ.savefig("plot_occ.pdf")

    sns.heatmap(all_symm_trafo_virt_ovlp, linewidth=0.5, ax=axs_virt[2], norm=LogNorm())

    axs_virt[0].set_title("Before sorting")
    axs_virt[1].set_title("After sorting")
    axs_virt[2].set_title("After symmetrization")

    fig_virt.savefig("plot_virt.pdf")

    return symm_eqv_mos, mo_coeff


def symmetrize_mos(
    mol: gto.Mole,
    mo_coeff: np.ndarray,
    nop: int,
    tot_symm_occ_blocks: List[List[int]],
    tot_symm_virt_blocks: List[List[int]],
    trafo_ao: np.ndarray,
    ncore: int,
    nocc: int,
) -> np.ndarray:
    """
    this function symmetrizes localized orbitals
    """
    # occupied - occupied block
    loc = symmetrize_eqv(
        mol, trafo_ao, tot_symm_occ_blocks, mo_coeff[:, ncore : min(mol.nelec)]
    )
    loc.verbose = 4
    mo_coeff[:, ncore : min(mol.nelec)] = loc.kernel()

    # virtual - virtual block
    loc = symmetrize_eqv(
        mol, trafo_ao, tot_symm_virt_blocks, mo_coeff[:, max(mol.nelec) :]
    )
    loc.verbose = 4
    mo_coeff[:, max(mol.nelec) :] = loc.kernel()

    symm_eqv_occ_mo: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
    symm_eqv_virt_mo: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]

    # loop over symmetry-invariant blocks
    for block in tot_symm_occ_blocks:

        symm_eqv_occ_mo = []

        # loop over symmetry operations
        for op in range(nop):

            # get mo coefficients for block
            block_mo_coeff = mo_coeff[:, ncore + np.array(block)]

            # transform mos
            op_block_mo_coeff = trafo_ao[op] @ block_mo_coeff

            # get unitary transformation
            symm_trafo = block_mo_coeff.T @ op_block_mo_coeff

            # init list of symmetry equivalent mos for this symmetry operation
            symm_eqv_occ_mo.append([])

            # lists of orbitals
            orbs = np.arange(len(block))
            trafo_orbs = np.arange(len(block))

            # init combination length
            comb_length = 1

            # loop over combinations of different length
            while comb_length <= len(block):

                # create identity matrix
                eye = np.eye(comb_length)

                # loop over combinations
                for comb1 in combinations(orbs, comb_length):

                    if any([orb not in orbs for orb in comb1]):
                        continue

                    orbs1 = np.array(comb1)

                    # loop over combinations
                    for comb2 in combinations(trafo_orbs, comb_length):

                        orbs2 = np.array(comb2)

                        # get transformation matrix between orbitals in orbitals in first
                        # and second combination
                        orb_trafo = symm_trafo[orbs1.reshape(-1, 1), orbs2]

                        # quantify deviation from symmetry by comparing to unitary matrix
                        deviation = (
                            np.linalg.norm(orb_trafo.T @ orb_trafo - eye)
                            / comb_length**2
                        )

                        # check if block vanishes
                        if deviation < MO_SYMM_TOL or comb_length == len(orbs):

                            # sets of orbitals are symmetry equivalent
                            symm_eqv_occ_mo[-1].append((comb1, comb2))

                            # remove orbitals
                            orbs = np.setdiff1d(orbs, comb1)
                            trafo_orbs = np.setdiff1d(trafo_orbs, comb2)

                            break

                # increment combination length
                comb_length += 1

            if orbs:

                symm_eqv_occ_mo[-1].append((tuple(orbs), tuple(trafo_orbs)))

        # occupied - occupied block
        loc = symmetrize_all(mol, trafo_ao, symm_eqv_occ_mo, block_mo_coeff)
        loc.conv_tol = 1e1 * SYMM_TOL
        loc.verbose = 4
        block_mo_coeff = loc.kernel()

    # loop over symmetry-invariant blocks
    for block in tot_symm_virt_blocks:

        symm_eqv_virt_mo = []

        # loop over symmetry operations
        for op in range(nop):

            # get mo coefficients for block
            block_mo_coeff = mo_coeff[:, nocc + np.array(block)]

            # transform mos
            op_block_mo_coeff = trafo_ao[op] @ block_mo_coeff

            # get unitary transformation
            symm_trafo = block_mo_coeff.T @ op_block_mo_coeff

            # init list of symmetry equivalent mos for this symmetry operation
            symm_eqv_virt_mo.append([])

            # lists of orbitals
            orbs = np.arange(len(block))
            trafo_orbs = np.arange(len(block))

            # init combination length
            comb_length = 1

            # loop over combinations of different length
            while comb_length <= len(block):

                # create identity matrix
                eye = np.eye(comb_length)

                # loop over combinations
                for comb1 in combinations(orbs, comb_length):

                    if any([orb not in orbs for orb in comb1]):
                        continue

                    orbs1 = np.array(comb1)

                    # loop over combinations
                    for comb2 in combinations(trafo_orbs, comb_length):

                        orbs2 = np.array(comb2)

                        # get transformation matrix between orbitals in orbitals in first
                        # and second combination
                        orb_trafo = symm_trafo[orbs1.reshape(-1, 1), orbs2]

                        # quantify deviation from symmetry by comparing to unitary matrix
                        deviation = (
                            np.linalg.norm(orb_trafo.T @ orb_trafo - eye)
                            / comb_length**2
                        )

                        # check if block vanishes
                        if deviation < MO_SYMM_TOL or comb_length == len(orbs):

                            # sets of orbitals are symmetry equivalent
                            symm_eqv_virt_mo[-1].append((comb1, comb2))

                            # remove orbitals
                            orbs = np.setdiff1d(orbs, comb1)
                            trafo_orbs = np.setdiff1d(trafo_orbs, comb2)

                            break

                # increment combination length
                comb_length += 1

            if orbs:

                symm_eqv_virt_mo[-1].append((tuple(orbs), tuple(trafo_orbs)))

        # virtual - virtual block
        loc = symmetrize_all(mol, trafo_ao, symm_eqv_virt_mo, block_mo_coeff)
        loc.conv_tol = 1e1 * SYMM_TOL
        loc.verbose = 4
        block_mo_coeff = loc.kernel()

    return mo_coeff


class symmetrize(soscf.ciah.CIAHOptimizer):
    r"""
    The symmetrization optimizer that minimizes blocks of the symmetry operation
    transformation matrix

    Args:
        mol : Mole object

    Attributes for symmetrize class:
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB. Default value equals to :class:`Mole.max_memory`.
        conv_tol : float
            Converge threshold. Default SYMM_TOL
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

    conv_tol = SYMM_TOL
    max_cycle = 100
    max_iters = 20
    max_stepsize = 0.05
    ah_trust_region = 3
    ah_start_tol = 1e9
    ah_max_cycle = 40
    ah_lindep = 1e-200

    def __init__(
        self,
        mol: gto.Mole,
        symm_ops: np.ndarray,
        symm_eqv_orbs: Union[
            List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]], List[List[int]]
        ],
        mo_coeff: np.ndarray,
    ):
        soscf.ciah.CIAHOptimizer.__init__(self)
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

    def kernel(
        self, callback: Optional[Callable] = None, verbose: Optional[int] = None
    ):
        from pyscf.tools import mo_mapping

        if self.mo_coeff.shape[1] <= 1:
            return self.mo_coeff

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.new_logger(self, verbose=verbose)

        conv_tol_grad = np.sqrt(self.conv_tol * 0.1)

        u0 = np.eye(self.mo_coeff.shape[1])

        rotaiter = soscf.ciah.rotate_orb_cc(self, u0, conv_tol_grad, verbose=log)
        u, _, stat = next(rotaiter)
        cput1 = log.timer("initializing CIAH", *cput0)

        tot_kf = stat.tot_kf
        tot_hop = stat.tot_hop
        conv = False
        e_last = 0
        for imacro in range(self.max_cycle):
            u0 = np.dot(u0, u)
            e, e_max = self.cost_function(u0)
            e_last, de = e, e - e_last

            log.info(
                "macro= %d  f(x)= %.14g  delta_f= %g  max(Gpq)= %g  %d KF %d Hx",
                imacro + 1,
                e,
                de,
                e_max,
                stat.tot_kf + 1,
                stat.tot_hop,
            )
            cput1 = log.timer("cycle= %d" % (imacro + 1), *cput1)

            if e_max < self.conv_tol:
                conv = True

            if callable(callback):
                callback(locals())

            if conv:
                break

            u, _, stat = rotaiter.send(u0)
            tot_kf += stat.tot_kf
            tot_hop += stat.tot_hop

        rotaiter.close()
        log.info(
            "macro X = %d  f(x)= %.14g  max(Gpq)= %g  %d intor %d KF %d Hx",
            imacro + 1,
            e,
            e_max,
            (imacro + 1) * 2,
            tot_kf + imacro + 1,
            tot_hop,
        )
        # Sort the symmetrized orbitals to make each orbital as close as
        # possible to the corresponding input orbitals
        sorted_idx = mo_mapping.mo_1to1map(u0)
        self.mo_coeff = np.dot(self.mo_coeff, u0[:, sorted_idx])
        return self.mo_coeff


class symmetrize_all(symmetrize):
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
                        "ss_ij": ss[incl.reshape(-1, 1), incl],
                        "ss_ia": ss[incl.reshape(-1, 1), excl],
                        "ss_ai": ss[excl.reshape(-1, 1), incl],
                        "ss_ab": ss[excl.reshape(-1, 1), excl],
                        "su_ia": su[incl.reshape(-1, 1), excl],
                        "su_ab": su[excl.reshape(-1, 1), excl],
                        "us_ij": us[incl.reshape(-1, 1), incl],
                        "us_ia": us[incl.reshape(-1, 1), excl],
                        "uu_ia": uu[incl.reshape(-1, 1), excl],
                    }
                )

                interm[-1][-1]["su_ia@su_ia.T"] = (
                    interm[-1][-1]["su_ia"] @ interm[-1][-1]["su_ia"].T
                )
                interm[-1][-1]["su_ab@su_ab.T"] = (
                    interm[-1][-1]["su_ab"] @ interm[-1][-1]["su_ab"].T
                )
                interm[-1][-1]["us_ij.T@us_ij"] = (
                    interm[-1][-1]["us_ij"].T @ interm[-1][-1]["us_ij"]
                )
                interm[-1][-1]["us_ia.T@us_ia"] = (
                    interm[-1][-1]["us_ia"].T @ interm[-1][-1]["us_ia"]
                )

        # calculate gradient
        g0 = np.zeros((norb, norb), dtype=np.float64)
        for op in range(len(self.symm_ops)):
            for orbset, (incl, excl) in enumerate(
                zip(self.incl_orbs[op], self.excl_orbs[op])
            ):

                su_ia = interm[op][orbset]["su_ia"]
                uu_ia = interm[op][orbset]["uu_ia"]
                us_ij = interm[op][orbset]["us_ij"]
                su_ab = interm[op][orbset]["su_ab"]
                us_ia = interm[op][orbset]["us_ia"]

                g0[incl.reshape(-1, 1), incl] += su_ia @ uu_ia.T
                g0[incl.reshape(-1, 1), excl] += us_ij.T @ uu_ia - uu_ia @ su_ab.T
                g0[excl.reshape(-1, 1), excl] += us_ia.T @ uu_ia

        g = self.pack_uniq_var(g0 - g0.T) * 2

        # calculate hessian diagonal
        h_diag0 = np.zeros((norb, norb), dtype=np.float64)
        for op in range(len(self.symm_ops)):
            for orbset, (incl, excl) in enumerate(
                zip(self.incl_orbs[op], self.excl_orbs[op])
            ):

                su_ia = interm[op][orbset]["su_ia"]
                us_ij = interm[op][orbset]["us_ij"]
                su_ab = interm[op][orbset]["su_ab"]
                uu_ia = interm[op][orbset]["uu_ia"]
                ss_ai = interm[op][orbset]["ss_ai"]
                us_ia = interm[op][orbset]["us_ia"]
                su_ia_su_iaT = interm[op][orbset]["su_ia@su_ia.T"]
                su_ab_su_abT = interm[op][orbset]["su_ab@su_ab.T"]
                us_ijT_us_ij = interm[op][orbset]["us_ij.T@us_ij"]
                us_iaT_us_ia = interm[op][orbset]["us_ia.T@us_ia"]

                h_diag0[incl.reshape(-1, 1), incl] += np.diag(su_ia_su_iaT)[
                    :, np.newaxis
                ]
                h_diag0[incl.reshape(-1, 1), excl] += (
                    np.diag(us_ijT_us_ij)[:, np.newaxis]
                    + np.diag(su_ab_su_abT)[np.newaxis, :]
                    - np.einsum("bb,jj->jb", su_ab, us_ij)
                    - 2 * np.einsum("jb,bj->jb", uu_ia, ss_ai)
                    - np.einsum("jj,bb->jb", us_ij, su_ab)
                )
                h_diag0[excl.reshape(-1, 1), excl] += np.diag(us_iaT_us_ia)[
                    :, np.newaxis
                ]

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

                    x_ij = x[incl.reshape(-1, 1), incl]
                    x_ia = x[incl.reshape(-1, 1), excl]
                    x_ab = x[excl.reshape(-1, 1), excl]

                    su_ia = interm[op][orbset]["su_ia"]
                    su_ab = interm[op][orbset]["su_ab"]
                    us_ij = interm[op][orbset]["us_ij"]
                    ss_ij = interm[op][orbset]["ss_ij"]
                    uu_ia = interm[op][orbset]["uu_ia"]
                    us_ia = interm[op][orbset]["us_ia"]
                    ss_ia = interm[op][orbset]["ss_ia"]
                    ss_ai = interm[op][orbset]["ss_ai"]
                    ss_ab = interm[op][orbset]["ss_ab"]

                    su_ia_su_iaT = interm[op][orbset]["su_ia@su_ia.T"]
                    su_ab_su_abT = interm[op][orbset]["su_ab@su_ab.T"]
                    us_ijT_us_ij = interm[op][orbset]["us_ij.T@us_ij"]
                    us_iaT_us_ia = interm[op][orbset]["us_ia.T@us_ia"]
                    us_ij_x_ia = us_ij @ x_ia
                    x_ia_su_ab = x_ia @ su_ab
                    x_iaT_uu_ia = x_ia.T @ uu_ia
                    x_ia_uu_iaT = x_ia @ uu_ia.T
                    x_ij_uu_ia = x_ij @ uu_ia
                    x_ijT_su_ia = x_ij.T @ su_ia

                    hx_ij = (
                        su_ia_su_iaT @ x_ij
                        + x_ia_su_ab @ su_ia.T
                        + su_ia @ us_ij_x_ia.T
                        + ss_ij @ x_ia_uu_iaT
                        + su_ia @ x_ab.T @ us_ia.T
                        + ss_ia @ x_ab @ uu_ia.T
                    )
                    hx0[incl.reshape(-1, 1), incl] += hx_ij
                    hx_ia = (
                        x_ij @ su_ia @ su_ab.T
                        + us_ij.T @ x_ijT_su_ia
                        + ss_ij.T @ x_ij_uu_ia
                        + x_ia @ su_ab_su_abT
                        + us_ijT_us_ij @ x_ia
                        - us_ij_x_ia @ su_ab.T
                        - x_ia_uu_iaT.T @ ss_ai.T
                        - us_ij.T @ x_ia_su_ab
                        - ss_ai.T @ x_iaT_uu_ia
                        + us_ij.T @ us_ia @ x_ab.T
                        + us_ia @ x_ab @ su_ab.T
                        + uu_ia @ x_ab.T @ ss_ab.T
                    )
                    hx0[incl.reshape(-1, 1), excl] += hx_ia
                    hx_ab = (
                        us_ia.T @ x_ijT_su_ia
                        + ss_ia.T @ x_ij_uu_ia
                        + us_ij_x_ia.T @ us_ia
                        + us_ia.T @ x_ia_su_ab
                        + ss_ab.T @ x_iaT_uu_ia
                        + us_iaT_us_ia @ x_ab
                    )
                    hx0[excl.reshape(-1, 1), excl] += hx_ab

            return DIFF * self.pack_uniq_var(hx0 - hx0.T) * 4

        return DIFF * g, h_op, DIFF * h_diag

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

                su_ia = su[op][incl.reshape(-1, 1), excl]
                uu_ia = uu[op][incl.reshape(-1, 1), excl]
                us_ij = us[op][incl.reshape(-1, 1), incl]
                su_ab = su[op][excl.reshape(-1, 1), excl]
                us_ia = us[op][incl.reshape(-1, 1), excl]

                g0[incl.reshape(-1, 1), incl] += su_ia @ uu_ia.T
                g0[incl.reshape(-1, 1), excl] += us_ij.T @ uu_ia - uu_ia @ su_ab.T
                g0[excl.reshape(-1, 1), excl] += us_ia.T @ uu_ia

        return DIFF * self.pack_uniq_var(g0 - g0.T) * 2

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

        return DIFF * p, g_max


class symmetrize_eqv(symmetrize):
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
        interm1: List[List[Dict[str, np.ndarray]]] = []
        interm2: List[Dict[str, np.ndarray]] = []

        for incl, excl in zip(self.incl_orbs, self.excl_orbs):

            n_incl = len(range(*incl.indices(norb)))
            n_excl = len(range(*excl.indices(norb)))

            interm2.append(
                {
                    "su_ia@su_ia.T": np.zeros((n_incl, n_incl), dtype=np.float64),
                    "su_ab@su_ia.T": np.zeros((n_excl, n_incl), dtype=np.float64),
                    "su_ab@su_ab.T": np.zeros((n_excl, n_excl), dtype=np.float64),
                    "us_ij.T@us_ij": np.zeros((n_incl, n_incl), dtype=np.float64),
                    "us_ij.T@us_ia": np.zeros((n_incl, n_excl), dtype=np.float64),
                    "us_ia.T@us_ia": np.zeros((n_excl, n_excl), dtype=np.float64),
                }
            )

        for op, op_mat in enumerate(self.symm_ops):

            uu = mo_coeff_u.T @ op_mat @ mo_coeff_u
            ss = mo_coeff_s.T @ op_mat @ mo_coeff_s
            us = mo_coeff_u.T @ op_mat @ mo_coeff_s
            su = mo_coeff_s.T @ op_mat @ mo_coeff_u

            interm1.append([])

            for orbset, (incl, excl) in enumerate(zip(self.incl_orbs, self.excl_orbs)):

                interm1[-1].append(
                    {
                        "ss_ij": ss[incl, incl],
                        "ss_ia": ss[incl, excl],
                        "ss_ai": ss[excl, incl],
                        "ss_ab": ss[excl, excl],
                        "su_ia": su[incl, excl],
                        "su_ab": su[excl, excl],
                        "us_ij": us[incl, incl],
                        "us_ia": us[incl, excl],
                        "uu_ia": uu[incl, excl],
                    }
                )

                interm2[orbset]["su_ia@su_ia.T"] += (
                    interm1[-1][-1]["su_ia"] @ interm1[-1][-1]["su_ia"].T
                )
                interm2[orbset]["su_ab@su_ia.T"] += (
                    interm1[-1][-1]["su_ab"] @ interm1[-1][-1]["su_ia"].T
                )
                interm2[orbset]["su_ab@su_ab.T"] += (
                    interm1[-1][-1]["su_ab"] @ interm1[-1][-1]["su_ab"].T
                )
                interm2[orbset]["us_ij.T@us_ij"] += (
                    interm1[-1][-1]["us_ij"].T @ interm1[-1][-1]["us_ij"]
                )
                interm2[orbset]["us_ij.T@us_ia"] += (
                    interm1[-1][-1]["us_ij"].T @ interm1[-1][-1]["us_ia"]
                )
                interm2[orbset]["us_ia.T@us_ia"] += (
                    interm1[-1][-1]["us_ia"].T @ interm1[-1][-1]["us_ia"]
                )

        # calculate gradient
        g0 = np.zeros((norb, norb), dtype=np.float64)
        for orbset, (incl, excl) in enumerate(zip(self.incl_orbs, self.excl_orbs)):

            for op in range(len(self.symm_ops)):

                su_ia = interm1[op][orbset]["su_ia"]
                uu_ia = interm1[op][orbset]["uu_ia"]
                us_ij = interm1[op][orbset]["us_ij"]
                su_ab = interm1[op][orbset]["su_ab"]
                us_ia = interm1[op][orbset]["us_ia"]

                g0[incl, incl] += su_ia @ uu_ia.T
                g0[incl, excl] += us_ij.T @ uu_ia - uu_ia @ su_ab.T
                g0[excl, excl] += us_ia.T @ uu_ia

        g = self.pack_uniq_var(g0 - g0.T) * 2

        # calculate hessian diagonal
        h_diag0 = np.zeros((norb, norb), dtype=np.float64)
        for orbset, (incl, excl) in enumerate(zip(self.incl_orbs, self.excl_orbs)):

            for op in range(len(self.symm_ops)):

                su_ia = interm1[op][orbset]["su_ia"]
                us_ij = interm1[op][orbset]["us_ij"]
                su_ab = interm1[op][orbset]["su_ab"]
                uu_ia = interm1[op][orbset]["uu_ia"]
                ss_ai = interm1[op][orbset]["ss_ai"]
                us_ia = interm1[op][orbset]["us_ia"]

                h_diag0[incl, excl] += (
                    -np.einsum("bb,jj->jb", su_ab, us_ij)
                    - 2 * np.einsum("jb,bj->jb", uu_ia, ss_ai)
                    - np.einsum("jj,bb->jb", us_ij, su_ab)
                )

            su_ia_su_iaT = interm2[orbset]["su_ia@su_ia.T"]
            su_ab_su_abT = interm2[orbset]["su_ab@su_ab.T"]
            us_ijT_us_ij = interm2[orbset]["us_ij.T@us_ij"]
            us_iaT_us_ia = interm2[orbset]["us_ia.T@us_ia"]

            h_diag0[incl, incl] += np.diag(su_ia_su_iaT)[:, np.newaxis]
            h_diag0[incl, excl] += (
                np.diag(us_ijT_us_ij)[:, np.newaxis]
                + np.diag(su_ab_su_abT)[np.newaxis, :]
            )
            h_diag0[excl, excl] += np.diag(us_iaT_us_ia)[:, np.newaxis]

        h_diag = self.pack_uniq_var(h_diag0 + h_diag0.T) * 2

        def h_op(x: np.ndarray):
            """
            this function calculates the matrix-vector product with some vector x
            """
            # calculate hx
            x = self.unpack_uniq_var(x)
            hx0 = np.zeros_like(x)
            for orbset, (incl, excl) in enumerate(zip(self.incl_orbs, self.excl_orbs)):

                x_ij = x[incl, incl]
                x_ia = x[incl, excl]
                x_ab = x[excl, excl]

                for op in range(len(self.symm_ops)):

                    su_ia = interm1[op][orbset]["su_ia"]
                    us_ij = interm1[op][orbset]["us_ij"]
                    ss_ij = interm1[op][orbset]["ss_ij"]
                    uu_ia = interm1[op][orbset]["uu_ia"]
                    us_ia = interm1[op][orbset]["us_ia"]
                    ss_ia = interm1[op][orbset]["ss_ia"]
                    su_ab = interm1[op][orbset]["su_ab"]
                    ss_ai = interm1[op][orbset]["ss_ai"]
                    ss_ab = interm1[op][orbset]["ss_ab"]

                    us_ij_x_ia = us_ij @ x_ia
                    x_ia_su_ab = x_ia @ su_ab
                    x_iaT_uu_ia = x_ia.T @ uu_ia
                    x_ia_uu_iaT = x_ia @ uu_ia.T
                    x_ij_uu_ia = x_ij @ uu_ia
                    x_ijT_su_ia = x_ij.T @ su_ia

                    hx0[incl, incl] += (
                        su_ia @ us_ij_x_ia.T
                        + ss_ij @ x_ia_uu_iaT
                        + su_ia @ x_ab.T @ us_ia.T
                        + ss_ia @ x_ab @ uu_ia.T
                    )
                    hx0[incl, excl] += (
                        us_ij.T @ x_ijT_su_ia
                        + ss_ij.T @ x_ij_uu_ia
                        - us_ij_x_ia @ su_ab.T
                        - x_ia_uu_iaT.T @ ss_ai.T
                        - us_ij.T @ x_ia_su_ab
                        - ss_ai.T @ x_iaT_uu_ia
                        + us_ia @ x_ab @ su_ab.T
                        + uu_ia @ x_ab.T @ ss_ab.T
                    )
                    hx0[excl, excl] += (
                        us_ia.T @ x_ijT_su_ia
                        + ss_ia.T @ x_ij_uu_ia
                        + us_ia.T @ x_ia_su_ab
                        + ss_ab.T @ x_iaT_uu_ia
                    )

                su_ia_su_iaT = interm2[orbset]["su_ia@su_ia.T"]
                su_ab_su_iaT = interm2[orbset]["su_ab@su_ia.T"]
                su_ab_su_abT = interm2[orbset]["su_ab@su_ab.T"]
                us_ijT_us_ij = interm2[orbset]["us_ij.T@us_ij"]
                us_ijT_us_ia = interm2[orbset]["us_ij.T@us_ia"]
                us_iaT_us_ia = interm2[orbset]["us_ia.T@us_ia"]

                hx0[incl, incl] += su_ia_su_iaT @ x_ij + x_ia @ su_ab_su_iaT
                hx0[incl, excl] += (
                    x_ij @ su_ab_su_iaT.T
                    + x_ia @ su_ab_su_abT
                    + us_ijT_us_ij @ x_ia
                    + us_ijT_us_ia @ x_ab.T
                )
                hx0[excl, excl] += x_ia.T @ us_ijT_us_ia + us_iaT_us_ia @ x_ab

            return DIFF * self.pack_uniq_var(hx0 - hx0.T) * 4

        return DIFF * g, h_op, DIFF * h_diag

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

                su_ia = su[op][incl, excl]
                uu_ia = uu[op][incl, excl]
                us_ij = us[op][incl, incl]
                su_ab = su[op][excl, excl]
                us_ia = us[op][incl, excl]

                g0[incl, incl] += su_ia @ uu_ia.T
                g0[incl, excl] += us_ij.T @ uu_ia - uu_ia @ su_ab.T
                g0[excl, excl] += us_ia.T @ uu_ia

        return DIFF * self.pack_uniq_var(g0 - g0.T) * 2

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

        return DIFF * p, g_max


def symm_trafo_ao(mol: gto.Mole, point_group: str, sao: np.ndarray) -> np.ndarray:
    """
    generates symmetry operation transformation matrix in orthogonal ao basis
    """
    # get atom coords
    coords = mol.atom_coords()

    # get symmetry origin and axes
    symm_orig, symm_axes = get_symm_coord(point_group, mol._atom, mol._basis)

    # shift coordinates to symmetry origin
    coords -= symm_orig

    # rotate coordinates to symmetry axes
    coords = (symm_axes.T @ coords.T).T

    # get Wigner D matrices to rotate aos from input coordinate system to symmetry axes
    Ds = symm.basis._ao_rotation_matrices(mol, symm_axes)

    # get different equivalent atom types
    atom_types = [
        np.array(atom_type)
        for atom_type in gto.mole.atom_types(mol._atom, mol.basis).values()
    ]

    # get ao offset for every atom
    _, _, ao_start_list, ao_stop_list = mol.offset_nr_by_atom().T

    # get ao shell offsets
    ao_loc = mol.ao_loc_nr()

    # get angular momentum for each shell
    l_shell = [mol.bas_angular(shell) for shell in range(mol.nbas)]

    # get list of all symmetry operation matrices for point group
    ops = get_symm_op_matrices(point_group, max(l_shell))

    # initialize atom indices permutation array
    permut_atom_idx = np.empty(mol.natm, dtype=np.int64)

    # initialize ao indices permutation array for every symmetry operation
    permut_ao_idx = np.empty((len(ops), mol.nao), dtype=np.int64)

    # initialize ao transformation matrices for different symmetry operations
    trafo_ao = np.zeros((len(ops), mol.nao, mol.nao), dtype=np.float64)

    # loop over symmetry operations
    for op, (cart_op_mat, sph_op_mats) in enumerate(ops):

        # loop over group of equivalent atoms with equivalent basis functions
        for atom_type in atom_types:

            # extract coordinates of atom type
            atom_coords = coords[atom_type]

            # get indices necessary to sort coords lexicographically
            lex_idx = symm.geom.argsort_coords(atom_coords)

            # sort coords lexicographically
            lex_coords = atom_coords[lex_idx]

            # get indices necessary to sort atom numbers
            sort_idx = np.argsort(lex_idx)

            # get new coordinates of atoms after applying symmetry operation
            new_atom_coords = (cart_op_mat.T @ atom_coords.T).T

            # get indices necessary to sort new coords lexicographically
            lex_idx = symm.geom.argsort_coords(new_atom_coords)

            # check whether rearranged new coords are the same as rearranged original
            # coords
            if not np.allclose(lex_coords, new_atom_coords[lex_idx], atol=COORD_TOL):
                raise PointGroupSymmetryError(
                    "Symmetry identical atoms not found. Please ensure coordinates "
                    "are symmetrical up to very high precision."
                )

            # reorder indices according to sort order of original indices
            op_atom_ids = lex_idx[sort_idx]

            # add atom permutations for atom type
            permut_atom_idx[atom_type] = atom_type[op_atom_ids]

        # loop over atoms
        for atom_id, permut_atom_id in enumerate(permut_atom_idx):

            # add ao permutations for atom
            permut_ao_idx[
                op, ao_start_list[atom_id] : ao_stop_list[atom_id]
            ] = np.arange(ao_start_list[permut_atom_id], ao_stop_list[permut_atom_id])

        # rotate symmetry operation matrices for spherical harmonics to original
        # coordinates and back
        rot_sph_op_mats = [
            rot_mat.T @ op_mat @ rot_mat for rot_mat, op_mat in zip(Ds, sph_op_mats)
        ]

        # loop over shells
        for shell, l in enumerate(l_shell):

            # loop over contracted basis functions in shell
            for bf in range(mol.bas_nctr(shell)):

                # get ao index range for contracted basis function
                ao_start = ao_loc[shell] + bf * rot_sph_op_mats[l].shape[1]
                ao_stop = ao_start + rot_sph_op_mats[l].shape[1]

                # insert transformation matrix
                trafo_ao[op, ao_start:ao_stop, ao_start:ao_stop] = rot_sph_op_mats[l]

        # permute aos
        trafo_ao[op] = trafo_ao[op, permut_ao_idx[op]]

        # transform to orthogonal ao basis
        trafo_ao[op] = sao @ trafo_ao[op]

    return trafo_ao
