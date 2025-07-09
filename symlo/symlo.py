#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main symlo module
"""

from __future__ import annotations

__author__ = "Jonas Greiner, Technical University of Denmark, Denmark"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Jonas Greiner"
__email__ = "jongr@dtu.dk"
__status__ = "Development"

import numpy as np
from pyscf import gto, symm
from pyscf.lib.exceptions import PointGroupSymmetryError
from pyscf.lib import logger
from typing import TYPE_CHECKING

from symlo.tools import (
    get_symm_op_matrices,
    get_symm_coord,
    get_mo_trafos,
    get_symm_inv_blocks,
    get_symm_unique_mos,
)
from symlo.symmetrization import SymCls_all, SymCls_eqv
from symlo.opentrustregion_interface import SymCls_all_OTR, SymCls_eqv_OTR

if TYPE_CHECKING:
    from typing import Tuple, List, Optional


COORD_TOL = 1.0e-14


def symmetrize_mos(
    mol: gto.Mole,
    mo_coeff: np.ndarray,
    point_group: str,
    verbose: Optional[int] = None,
    max_cycle: int = 100,
    conv_tol: float = 1e-13,
    inv_block_thresh: float = 0.3,
    symm_eqv_thresh: float = 0.3,
    heatmap: bool = False,
    start_idx: int = 0,
    backend: str = "pyscf",
) -> Tuple[
    List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
    np.ndarray,
    List[List[int]],
    List[List[List[Tuple[int, int]]]],
]:
    """
    returns an array of permutations of symmetry equivalent orbitals for each
    symmetry operation and symmetrized orbitals
    """
    # set verbosity if it is not set
    if verbose is None:
        verbose = mol.verbose

    # initialize logger
    log = logger.new_logger(verbose=verbose)

    # set backend
    if backend == "pyscf":
        symcls_eqv = SymCls_eqv
        symcls_all = SymCls_all
    elif backend == "opentrustregion":
        symcls_eqv = SymCls_eqv_OTR
        symcls_all = SymCls_all_OTR

    # convert point group to standard symbol
    point_group = symm.std_symb(point_group)

    # get number of orbitals
    norb = mo_coeff.shape[1]

    # get ao overlap matrix
    sao = mol.intor("int1e_ovlp")

    # get symmetry transformation matrices in orthogonal ao basis
    trafo_ao = get_symm_trafo_ao(mol, point_group, sao)

    # get total number of symmetry operations
    nop = trafo_ao.shape[0]

    # transform mo coefficients
    op_mo_coeff = trafo_ao @ mo_coeff

    # get overlap between mos and transformed mos
    trafo_ovlp = np.einsum("ij,nik->njk", mo_coeff, op_mo_coeff)

    # get sum of overlap for all symmetry operations and normalize (the individual
    # matrices for each symmetry operation are not symmetric because symmetry
    # operations are not necessarily unitary but their sum is because a group includes
    # an inverse element for every symmetry operation)
    all_symm_trafo_ovlp = np.sum(np.abs(trafo_ovlp), axis=0) / nop

    # get blocks that are invariant with respect to all symmetry operations
    tot_symm_blocks, reorder = get_symm_inv_blocks(
        all_symm_trafo_ovlp, inv_block_thresh
    )

    # log data
    log.info("Symmetry-invariant orbital blocks:")
    log.info(
        "\n".join(
            [str([start_idx + orb for orb in block]) for block in tot_symm_blocks]
        )
    )

    # reorder mo coefficients
    symm_mo_coeff = mo_coeff[:, reorder]

    symm_eqv_mos: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = [
        [] for op in range(nop)
    ]

    # symmetrize with respect to symmetry-invariant blocks
    symm_inv = symcls_eqv(mol, trafo_ao, tot_symm_blocks, symm_mo_coeff)
    symm_inv.max_cycle = max_cycle
    symm_inv.verbose = verbose
    symm_inv.conv_tol = conv_tol
    symm_mo_coeff, _, _ = symm_inv.kernel()

    # initialize symmetry-invariant blocks and cyclic sets within blocks
    blocks: List[List[int]] = []
    blocks_cyclic_sets: List[List[List[Tuple[int, int]]]] = []

    # loop over symmetry-invariant blocks
    for block in tot_symm_blocks:
        # set threshold
        thresh = symm_eqv_thresh

        # symmetrize orbitals until successful completion
        success = False
        while not success:
            try:
                # detect symmetry-equivalent orbitals
                symm_eqv_mo = detect_eqv_symm(
                    symm_mo_coeff[:, block], trafo_ao, thresh, nop, True
                )

                # symmetrize block
                symm_block = symcls_all(
                    mol, trafo_ao, symm_eqv_mo, symm_mo_coeff[:, block]
                )
                symm_block.verbose = verbose
                symm_block.max_cycle = max_cycle
                symm_block.conv_tol = 1e1 * conv_tol
                temp_mo_coeff, _, g_max = symm_block.kernel()

                # symmetrization succeeded
                success = True

            except RuntimeError:
                thresh += (1 - thresh) / 2
                log.info(f"Symmetrization failed, threshold increased to: {thresh}")

        # set symmetrized orbitals
        symm_mo_coeff[:, block] = temp_mo_coeff

        # append blocks shifted by starting index
        blocks.append([start_idx + orb for orb in block])

        # get cyclic sets
        blocks_cyclic_sets.append([])
        for op in range(nop):
            # add equivalent orbitals
            symm_eqv_mos[op].extend(
                [
                    (
                        tuple(start_idx + block[orb] for orb in orb_comb[0]),
                        tuple(start_idx + block[orb] for orb in orb_comb[1]),
                    )
                    for orb_comb in symm_eqv_mo[op]
                ]
            )

            symm_eqv_mo_list = symm_eqv_mo[op].copy()

            blocks_cyclic_sets[-1].append([])

            # determine number of cyclic sets
            while len(symm_eqv_mo_list):
                set_norb = len(symm_eqv_mo_list[0][0])
                idx = 0
                set_size = 1
                start_tuple = symm_eqv_mo_list[0][0]
                prev_tuple = symm_eqv_mo_list[0][1]
                del symm_eqv_mo_list[0]
                while start_tuple != prev_tuple:
                    for idx in range(len(symm_eqv_mo_list)):
                        if symm_eqv_mo_list[idx][0] == prev_tuple:
                            break
                    else:
                        log.error(
                            "Could not find consistent cyclic symmetry relationships"
                        )
                        raise RuntimeError
                    set_size += 1
                    prev_tuple = symm_eqv_mo_list[idx][1]
                    del symm_eqv_mo_list[idx]
                blocks_cyclic_sets[-1][-1].append((set_norb, set_size))
            blocks_cyclic_sets[-1][-1].sort()

    # check if input files for heatmap should be printed
    if heatmap:
        # reorder overlap matrix
        sort_all_symm_ovlp = all_symm_trafo_ovlp[reorder.reshape(-1, 1), reorder]

        # transform mo coefficients
        op_mo_coeff = trafo_ao @ symm_mo_coeff

        # get overlap between mos and transformed mos
        new_ovlp = np.einsum("ij,nik->njk", symm_mo_coeff, op_mo_coeff)

        # get sum of overlap for all symmetry operations and normalize
        new_all_symm_ovlp = np.sum(np.abs(new_ovlp), axis=0) / nop

        np.save("overlap_before.npy", all_symm_trafo_ovlp)
        np.save("overlap_sorted.npy", sort_all_symm_ovlp)
        np.save("overlap_after.npy", new_all_symm_ovlp)

    symm_unique_mos = get_symm_unique_mos(symm_eqv_mos, norb, start_idx)

    # get number of symmetry-unique mos
    nunique = len(symm_unique_mos)

    log.info(f"\n\nTotal number of orbitals: {norb}")
    log.info(f"Number of symmetry-unique orbitals: {nunique}\n\n")

    return symm_eqv_mos, symm_mo_coeff, blocks, blocks_cyclic_sets


def detect_mo_symm(
    mol: gto.Mole,
    mo_coeff: np.ndarray,
    point_group: str,
    verbose: Optional[int] = None,
    inv_block_thresh: float = 0.3,
    symm_eqv_thresh: float = 0.3,
    start_idx: int = 0,
) -> List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    """
    returns an array of permutations of symmetry equivalent orbitals for each
    symmetry operation
    """
    # set verbosity if it is not set
    if verbose is None:
        verbose = mol.verbose

    # initialize logger
    log = logger.new_logger(verbose=verbose)

    # convert point group to standard symbol
    point_group = symm.std_symb(point_group)

    # get number of orbitals
    norb = mo_coeff.shape[1]

    # get ao overlap matrix
    sao = mol.intor("int1e_ovlp")

    # get symmetry transformation matrices in orthogonal ao basis
    trafo_ao = get_symm_trafo_ao(mol, point_group, sao)

    # get total number of symmetry operations
    nop = trafo_ao.shape[0]

    # transform mo coefficients
    op_mo_coeff = trafo_ao @ mo_coeff

    # get overlap between mos and transformed mos
    trafo_ovlp = np.einsum("ij,nik->njk", mo_coeff, op_mo_coeff)

    # get sum of overlap for all symmetry operations and normalize (the individual
    # matrices for each symmetry operation are not symmetric because symmetry
    # operations are not necessarily unitary but their sum is because a group includes
    # an inverse element for every symmetry operation)
    all_symm_trafo_ovlp = np.sum(np.abs(trafo_ovlp), axis=0) / nop

    # get blocks that are invariant with respect to all symmetry operations
    tot_symm_blocks, reorder = get_symm_inv_blocks(
        all_symm_trafo_ovlp, inv_block_thresh
    )

    # log data
    log.info("Symmetry-invariant orbital blocks:")
    log.info(
        "\n".join(
            [
                str([start_idx + orb for orb in np.sort(reorder[block])])
                for block in tot_symm_blocks
            ]
        )
    )

    symm_eqv_mos: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = [
        [] for op in range(nop)
    ]

    # loop over symmetry-invariant blocks
    for block in tot_symm_blocks:
        # set threshold
        thresh = symm_eqv_thresh

        # detect symmetry-equivalent orbitals until successful completion
        success = False
        while not success:
            try:
                symm_eqv_mo = detect_eqv_symm(
                    mo_coeff[:, reorder[block]], trafo_ao, thresh, nop, True
                )
                success = True
            except RuntimeError:
                thresh += (1 - thresh) / 2
                log.info(f"Symmetry detection failed, threshold increased to: {thresh}")

        # add equivalent orbitals
        for op in range(nop):
            symm_eqv_mos[op].extend(
                [
                    (
                        tuple(
                            start_idx + reorder[block][orb].item()
                            for orb in orb_comb[0]
                        ),
                        tuple(
                            start_idx + reorder[block][orb].item()
                            for orb in orb_comb[1]
                        ),
                    )
                    for orb_comb in symm_eqv_mo[op]
                ]
            )

    symm_unique_mos = get_symm_unique_mos(symm_eqv_mos, norb, start_idx)

    # get number of symmetry-unique mos
    nunique = len(symm_unique_mos)

    log.info(f"\n\nTotal number of orbitals: {norb}")
    log.info(f"Number of symmetry-unique orbitals: {nunique}\n\n")

    return symm_eqv_mos


def get_symm_trafo_ao(mol: gto.Mole, point_group: str, sao: np.ndarray) -> np.ndarray:
    """
    this function generates a symmetry operation transformation matrix in the
    orthogonal ao basis
    """
    # get atom coords
    coords = mol.atom_coords()

    # get symmetry origin and axes
    symm_orig, symm_axes = get_symm_coord(point_group, mol._atom, mol._basis)

    # shift coordinates to symmetry origin
    coords -= symm_orig

    # rotate coordinates to symmetry axes
    coords = (symm_axes @ coords.T).T

    # get Wigner D matrices to rotate aos from symmetry axes to input coordinate system
    Ds = symm.basis._momentum_rotation_matrices(mol, symm_axes)

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
            new_atom_coords = (cart_op_mat @ atom_coords.T).T

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
            permut_ao_idx[op, ao_start_list[atom_id] : ao_stop_list[atom_id]] = (
                np.arange(ao_start_list[permut_atom_id], ao_stop_list[permut_atom_id])
            )

        # create combined rotation matrix that rotates AOs to symmetry axes, performs a
        # symmetry operation and rotates AOs back to original axes
        rot_sph_op_mats = [
            rot_mat @ op_mat @ rot_mat.T for rot_mat, op_mat in zip(Ds, sph_op_mats)
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
        trafo_ao[op] = trafo_ao[op, :, permut_ao_idx[op]]

        # transform to orthogonal ao basis
        trafo_ao[op] = sao @ trafo_ao[op]

    return trafo_ao


def detect_eqv_symm(
    mo_coeff: np.ndarray,
    trafo_ao: np.ndarray,
    symm_eqv_tol: float,
    nop: int,
    thresh: bool,
) -> List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    """
    this function detects symmetry-equivalent orbitals
    """
    # initialize list of symmetry-equivalent mos for every symmetry operation
    symm_eqv_mo = []

    # loop over symmetry operations
    for op in range(nop):
        # transform mos
        op_mo_coeff = trafo_ao[op] @ mo_coeff

        # get overlap of mos and transformed mos
        symm_trafo_ovlp = mo_coeff.T @ op_mo_coeff

        # add list of symmetry equivalent mos for this symmetry operation
        symm_eqv_mo.append(
            get_mo_trafos(symm_trafo_ovlp, mo_coeff.shape[1], symm_eqv_tol, thresh)
        )

    return symm_eqv_mo
