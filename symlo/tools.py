#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools module
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

try:
    from scipy.sparse import csr_array
except ImportError:
    from scipy.sparse import csr_matrix as csr_array
from math import sin, cos, pi, sqrt
from pyscf import symm, lib
from pyscf.lib.exceptions import PointGroupSymmetryError
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from typing import Tuple, Dict, List, Union, Set


def get_symm_op_matrices(
    point_group: str, l_max: int
) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    this function generates all cartesian and spherical symmetry operation matrices for
    a given point group
    """
    symm_ops = [_ident_matrix(l_max)]

    # 3D rotation group
    if point_group == "SO(3)":

        # same-atom symmetries are currently not exploited
        pass

    # proper cyclic groups Cn
    elif point_group[0] == "C" and point_group[1:].isnumeric():

        tot_main_rot = int(point_group[1:])

        # Cn
        for i in range(1, tot_main_rot):

            symm_ops.append(
                _rot_matrix(np.array([0.0, 0.0, 1.0]), i * 2 * pi / tot_main_rot, l_max)
            )

    # improper cyclic group Ci
    elif point_group == "Ci":

        # i
        symm_ops.append(_inv_matrix(l_max))

    # improper cyclic group Cs
    elif point_group == "Cs":

        # sigma_h
        symm_ops.append(_reflect_matrix(np.array([0.0, 0.0, 1.0]), l_max))

    # improper cyclic group Sn
    elif point_group[0] == "S":

        tot_main_rot = int(point_group[1:])

        # Cn, Sn and i
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            if i % 2 == 0:
                symm_ops.append(
                    _rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                )
            else:
                if rot_angle == pi:
                    symm_ops.append(_inv_matrix(l_max))
                else:
                    symm_ops.append(
                        _rot_reflect_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                    )

    # dihedral groups Dn
    elif point_group[0] == "D" and point_group[1:].isnumeric():

        tot_main_rot = int(point_group[1:])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(_rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # C2
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * pi
            symm_ops.append(
                _rot_matrix(np.array([cos(theta), sin(theta), 0.0]), pi, l_max)
            )

    # Dnh
    elif point_group[0] == "D" and point_group[-1] == "h":

        # treat Dooh as D2h because same-atom symmetries are currently not exploited
        if point_group[1:-1] == "oo":
            tot_main_rot = 2
        else:
            tot_main_rot = int(point_group[1:-1])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(_rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # C2
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * pi
            symm_ops.append(
                _rot_matrix(np.array([cos(theta), sin(theta), 0.0]), pi, l_max)
            )

        # Sn and i
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            if rot_angle == pi:
                symm_ops.append(_inv_matrix(l_max))
            else:
                symm_ops.append(
                    _rot_reflect_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                )

        # sigma_h
        symm_ops.append(_reflect_matrix(np.array([0.0, 0.0, 1.0]), l_max))

        # sigma_v and sigma_d
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * pi + pi / 2
            symm_ops.append(
                _reflect_matrix(np.array([cos(theta), sin(theta), 0.0]), l_max)
            )

    # Dnd
    elif point_group[0] == "D" and point_group[-1] == "d":

        tot_main_rot = int(point_group[1:-1])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(_rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # C2
        for i in range(0, tot_main_rot):
            theta = i / tot_main_rot * pi
            symm_ops.append(
                _rot_matrix(np.array([cos(theta), sin(theta), 0.0]), pi, l_max)
            )

        # S_2n
        for i in range(0, tot_main_rot):
            rot_angle = ((2 * i + 1) / tot_main_rot) * pi
            if rot_angle == pi:
                symm_ops.append(_inv_matrix(l_max))
            else:
                symm_ops.append(
                    _rot_reflect_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                )

        # sigma_d
        for i in range(0, tot_main_rot):
            theta = (2 * i + 1) / (2 * tot_main_rot) * pi + pi / 2
            symm_ops.append(
                _reflect_matrix(np.array([cos(theta), sin(theta), 0.0]), l_max)
            )

    # Cnv
    elif point_group[0] == "C" and point_group[-1] == "v":

        # treat Coov as C2v because same-atom symmetries are currently not exploited
        if point_group[1:-1] == "oo":
            tot_main_rot = 2
        else:
            tot_main_rot = int(point_group[1:-1])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(_rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # sigma_v and sigma_d
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * pi
            symm_ops.append(
                _reflect_matrix(np.array([cos(theta), sin(theta), 0.0]), l_max)
            )

    # Cnh
    elif point_group[0] == "C" and point_group[-1] == "h":

        tot_main_rot = int(point_group[1:-1])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(_rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # Sn and i
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            if rot_angle == pi:
                symm_ops.append(_inv_matrix(l_max))
            else:
                symm_ops.append(
                    _rot_reflect_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                )

        # sigma_h
        symm_ops.append(_reflect_matrix(np.array([0.0, 0.0, 1.0]), l_max))

    # cubic group O
    elif point_group == "O":

        corners, edges, surfaces = _cubic_coords()

        # C3
        for coord in corners:
            symm_ops.append(_rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(_rot_matrix(coord, 4 * pi / 3, l_max))

        # C4
        tot_n_rot = 4
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in surfaces:
                symm_ops.append(_rot_matrix(coord, rot_angle, l_max))

        # C2
        for coord in edges:
            symm_ops.append(_rot_matrix(coord, pi, l_max))

    # cubic group T
    elif point_group == "T":

        corners, edges, surfaces = _cubic_coords()

        # C2
        for coord in surfaces:
            symm_ops.append(_rot_matrix(coord, pi, l_max))

        # C3
        for coord in corners:
            symm_ops.append(_rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(_rot_matrix(coord, 4 * pi / 3, l_max))

    # cubic group Oh
    elif point_group == "Oh":

        corners, edges, surfaces = _cubic_coords()

        # C3
        for coord in corners:
            symm_ops.append(_rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(_rot_matrix(coord, 4 * pi / 3, l_max))

        # C4
        tot_n_rot = 4
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in surfaces:
                symm_ops.append(_rot_matrix(coord, rot_angle, l_max))

        # C2
        for coord in edges:
            symm_ops.append(_rot_matrix(coord, pi, l_max))

        # i
        symm_ops.append(_inv_matrix(l_max))

        # sigma
        for coord in surfaces:
            symm_ops.append(_reflect_matrix(coord, l_max))

        # S6
        for coord in corners:
            symm_ops.append(_rot_reflect_matrix(coord, pi / 3, l_max))
            symm_ops.append(_rot_reflect_matrix(coord, 5 * pi / 3, l_max))

        # S4
        tot_n_rot = 4
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            if rot_angle != pi:
                for coord in surfaces:
                    symm_ops.append(_rot_reflect_matrix(coord, rot_angle, l_max))

        # sigma_d
        for coord in edges:
            symm_ops.append(_reflect_matrix(coord, l_max))

    # cubic group Th
    elif point_group == "Th":

        corners, edges, surfaces = _cubic_coords()

        # C2
        for coord in surfaces:
            symm_ops.append(_rot_matrix(coord, pi, l_max))

        # C3
        for coord in corners:
            symm_ops.append(_rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(_rot_matrix(coord, 4 * pi / 3, l_max))

        # i
        symm_ops.append(_inv_matrix(l_max))

        # sigma
        for coord in surfaces:
            symm_ops.append(_reflect_matrix(coord, l_max))

        # S6
        for coord in corners:
            symm_ops.append(_rot_reflect_matrix(coord, pi / 3, l_max))
            symm_ops.append(_rot_reflect_matrix(coord, 5 * pi / 3, l_max))

    # cubic group Td
    elif point_group == "Td":

        corners, edges, surfaces = _cubic_coords()

        # C2
        for coord in surfaces:
            symm_ops.append(_rot_matrix(coord, pi, l_max))

        # C3
        for coord in corners:
            symm_ops.append(_rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(_rot_matrix(coord, 4 * pi / 3, l_max))

        # S4
        for coord in surfaces:
            symm_ops.append(_rot_reflect_matrix(coord, pi / 2, l_max))
            symm_ops.append(_rot_reflect_matrix(coord, 3 * pi / 2, l_max))

        # sigma_d
        for coord in edges:
            symm_ops.append(_reflect_matrix(coord, l_max))

    # icosahedral group I
    elif point_group == "I":

        corners, edges, surfaces = _icosahedric_coords()

        # C5
        tot_n_rot = 5
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in corners:
                symm_ops.append(_rot_matrix(coord, rot_angle, l_max))

        # C3
        tot_n_rot = 3
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in surfaces:
                symm_ops.append(_rot_matrix(coord, rot_angle, l_max))

        # C2
        for coord in edges:
            symm_ops.append(_rot_matrix(coord, pi, l_max))

    # icosahedral group Ih
    elif point_group == "Ih":

        corners, edges, surfaces = _icosahedric_coords()

        # C5
        tot_n_rot = 5
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in corners:
                symm_ops.append(_rot_matrix(coord, rot_angle, l_max))

        # C3
        tot_n_rot = 3
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in surfaces:
                symm_ops.append(_rot_matrix(coord, rot_angle, l_max))

        # C2
        for coord in edges:
            symm_ops.append(_rot_matrix(coord, pi, l_max))

        # i
        symm_ops.append(_inv_matrix(l_max))

        # S10
        tot_main_rot = 10
        for i in range(0, tot_main_rot // 2):
            rot_angle = ((2 * i + 1) / tot_main_rot) * 2 * pi
            if rot_angle != pi:
                for coord in corners:
                    symm_ops.append(_rot_reflect_matrix(coord, rot_angle, l_max))

        # S6
        tot_main_rot = 6
        for i in range(0, tot_main_rot // 2):
            rot_angle = ((2 * i + 1) / tot_main_rot) * 2 * pi
            if rot_angle != pi:
                for coord in surfaces:
                    symm_ops.append(_rot_reflect_matrix(coord, rot_angle, l_max))

        # sigma
        for coord in edges:
            symm_ops.append(_reflect_matrix(coord, l_max))

    else:

        raise PointGroupSymmetryError("Unknown Point Group.")

    return symm_ops


def get_symm_coord(
    point_group: str,
    atoms: List[List[Union[str, float]]],
    basis: Dict[str, List[List[Union[int, List[float]]]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function determines the charge center and symmetry axes for a given point group
    """
    # initialize symmetry object
    rawsys = symm.SymmSys(atoms, basis)

    # determine charge center of molecule
    charge_center = lib.parameters.BOHR * rawsys.charge_center

    # initialize boolean for correct point group
    correct_symm = False

    # 3D rotation group
    if point_group == "SO3":

        symm_axes = np.eye(3)

    # proper cyclic groups Cn
    elif point_group[0] == "C" and point_group[1:].isnumeric():

        if point_group[1:] == 1:

            correct_symm = True
            symm_axes = np.eye(3)

        else:

            tot_main_rot = int(point_group[1:])

            possible_cn = rawsys.search_possible_rotations()
            for zaxis, n in possible_cn:
                if n == tot_main_rot and rawsys.has_rotation(zaxis, n):
                    correct_symm = True
                    for axis in np.eye(3):
                        if not symm.parallel_vectors(axis, zaxis):
                            symm_axes = symm.geom._make_axes(zaxis, axis)
                            break
                    symm_axes = symm.geom._refine(symm_axes)
                    break

    # improper cyclic group Ci
    elif point_group == "Ci":

        if rawsys.has_icenter():
            correct_symm = True
            symm_axes = np.eye(3)

    # improper cyclic group Cs
    elif point_group == "Cs":

        mirror = rawsys.search_mirrorx(None, 1)

        if mirror is not None:
            correct_symm = True
            symm_axes = symm.geom._make_axes(mirror, np.array((1.0, 0.0, 0.0)))

    # improper cyclic group Sn
    elif point_group[0] == "S":

        tot_main_rot = int(point_group[1:])

        possible_cn = rawsys.search_possible_rotations()
        for zaxis, n in possible_cn:
            if 2 * n == tot_main_rot and rawsys.has_rotation(zaxis, n):
                for axis in np.eye(3):
                    if not symm.parallel_vectors(axis, zaxis):
                        symm_axes = symm.geom._make_axes(zaxis, axis)
                        break
                if rawsys.has_improper_rotation(symm_axes[2], n):
                    correct_symm = True
                    symm_axes = symm.geom._refine(symm_axes)
                    break

    # dihedral groups Dn
    elif point_group[0] == "D" and point_group[1:].isnumeric():

        tot_main_rot = int(point_group[1:])

        possible_cn = rawsys.search_possible_rotations()
        for zaxis, n in possible_cn:
            if n == tot_main_rot and rawsys.has_rotation(zaxis, n):
                c2x = rawsys.search_c2x(zaxis, n)
                if c2x is not None:
                    correct_symm = True
                    symm_axes = symm.geom._refine(symm.geom._make_axes(zaxis, c2x))
                    break

    # Dnh
    elif point_group[0] == "D" and point_group[-1] == "h":

        if point_group[1:-1] == "oo":

            w1, u1 = rawsys.cartesian_tensor(1)

            if (
                np.allclose(w1[:2], 0, atol=symm.TOLERANCE / np.sqrt(1 + len(atoms)))
                and rawsys.has_icenter()
            ):
                correct_symm = True
                symm_axes = u1.T

        else:

            tot_main_rot = int(point_group[1:-1])

            possible_cn = rawsys.search_possible_rotations()
            for zaxis, n in possible_cn:
                if n == tot_main_rot and rawsys.has_rotation(zaxis, n):
                    c2x = rawsys.search_c2x(zaxis, n)
                    if c2x is not None:
                        symm_axes = symm.geom._make_axes(zaxis, c2x)
                        if rawsys.has_mirror(symm_axes[2]):
                            correct_symm = True
                            symm_axes = symm.geom._refine(symm_axes)
                            break

    # Dnd
    elif point_group[0] == "D" and point_group[-1] == "d":

        tot_main_rot = int(point_group[1:-1])

        possible_cn = rawsys.search_possible_rotations()
        for zaxis, n in possible_cn:
            if n == tot_main_rot and rawsys.has_rotation(zaxis, n):
                c2x = rawsys.search_c2x(zaxis, n)
                if c2x is not None:
                    symm_axes = symm.geom._make_axes(zaxis, c2x)
                    if rawsys.has_improper_rotation(symm_axes[2], n):
                        correct_symm = True
                        symm_axes = symm.geom._refine(symm_axes)
                        break

    # Cnv
    elif point_group[0] == "C" and point_group[-1] == "v":

        if point_group[1:-1] == "oo":

            w1, u1 = rawsys.cartesian_tensor(1)

            if np.allclose(w1[:2], 0, atol=symm.TOLERANCE / np.sqrt(1 + len(atoms))):
                correct_symm = True
                symm_axes = u1.T

        else:

            tot_main_rot = int(point_group[1:-1])

            possible_cn = rawsys.search_possible_rotations()
            for zaxis, n in possible_cn:
                if n == tot_main_rot and rawsys.has_rotation(zaxis, n):
                    mirrorx = rawsys.search_mirrorx(zaxis, n)
                    if mirrorx is not None:
                        correct_symm = True
                        symm_axes = symm.geom._refine(
                            symm.geom._make_axes(zaxis, mirrorx)
                        )
                        break

    # Cnh
    elif point_group[0] == "C" and point_group[-1] == "h":

        tot_main_rot = int(point_group[1:-1])

        possible_cn = rawsys.search_possible_rotations()
        for zaxis, n in possible_cn:
            if n == tot_main_rot and rawsys.has_rotation(zaxis, n):
                for axis in np.eye(3):
                    if not symm.parallel_vectors(axis, zaxis):
                        symm_axes = symm.geom._make_axes(zaxis, axis)
                        break
                if rawsys.has_mirror(symm_axes[2]):
                    correct_symm = True
                    symm_axes = symm.geom._refine(symm_axes)
                    break

    # cubic group O
    elif point_group == "O":

        possible_cn = rawsys.search_possible_rotations()
        c4_axes = [c4 for c4, n in possible_cn if n == 4 and rawsys.has_rotation(c4, 4)]

        c4_axes = _oct_c4(c4_axes[0], c4_axes[1])

        if all([rawsys.has_rotation(c4, 4) for c4 in c4_axes]):
            correct_symm = True
            symm_axes = symm.geom._refine(symm.geom._make_axes(c4_axes[0], c4_axes[1]))

    # cubic group T
    elif point_group == "T":

        possible_cn = rawsys.search_possible_rotations()
        c3_axes = [c3 for c3, n in possible_cn if n == 3 and rawsys.has_rotation(c3, 3)]

        c3_axes = _tet_c3(c3_axes[0], c3_axes[1])

        if all([rawsys.has_rotation(c3, 3) for c3 in c3_axes]):
            correct_symm = True
            symm_axes = symm.geom._refine(
                symm.geom._make_axes(c3_axes[0] + c3_axes[1], c3_axes[0] + c3_axes[2])
            )

    # cubic group Oh
    elif point_group == "Oh":

        possible_cn = rawsys.search_possible_rotations()
        c4_axes = [c4 for c4, n in possible_cn if n == 4 and rawsys.has_rotation(c4, 4)]

        c4_axes = _oct_c4(c4_axes[0], c4_axes[1])

        if all([rawsys.has_rotation(c4, 4) for c4 in c4_axes]) and rawsys.has_icenter():
            correct_symm = True
            symm_axes = symm.geom._refine(symm.geom._make_axes(c4_axes[0], c4_axes[1]))

    # cubic group Th
    elif point_group == "Th":

        possible_cn = rawsys.search_possible_rotations()
        c3_axes = [c3 for c3, n in possible_cn if n == 3 and rawsys.has_rotation(c3, 3)]

        c3_axes = _tet_c3(c3_axes[0], c3_axes[1])

        if all([rawsys.has_rotation(c3, 3) for c3 in c3_axes]) and rawsys.has_icenter():
            correct_symm = True
            symm_axes = symm.geom._refine(
                symm.geom._make_axes(c3_axes[0] + c3_axes[1], c3_axes[0] + c3_axes[2])
            )

    # cubic group Td
    elif point_group == "Td":

        possible_cn = rawsys.search_possible_rotations()
        c3_axes = [c3 for c3, n in possible_cn if n == 3 and rawsys.has_rotation(c3, 3)]

        c3_axes = _tet_c3(c3_axes[0], c3_axes[1])

        if all([rawsys.has_rotation(c3, 3) for c3 in c3_axes]) and rawsys.has_mirror(
            np.cross(c3_axes[0], c3_axes[1])
        ):
            correct_symm = True
            symm_axes = symm.geom._refine(
                symm.geom._make_axes(c3_axes[0] + c3_axes[1], c3_axes[0] + c3_axes[2])
            )

    # icosahedral group I
    elif point_group == "I":

        possible_cn = rawsys.search_possible_rotations()
        c5_axes = [c5 for c5, n in possible_cn if n == 5 and rawsys.has_rotation(c5, 5)]

        c5_axes = _ico_c5(c5_axes[0], c5_axes[1])

        if all([rawsys.has_rotation(c5, 5) for c5 in c5_axes]):
            correct_symm = True
            symm_axes = symm.geom._refine(
                symm.geom._make_axes(c5_axes[0] + c5_axes[1], c5_axes[2] - c5_axes[5])
            )

    # icosahedral group Ih
    elif point_group == "Ih":

        possible_cn = rawsys.search_possible_rotations()
        c5_axes = [c5 for c5, n in possible_cn if n == 5 and rawsys.has_rotation(c5, 5)]

        c5_axes = _ico_c5(c5_axes[0], c5_axes[1])

        if all([rawsys.has_rotation(c5, 5) for c5 in c5_axes]) and rawsys.has_icenter():
            correct_symm = True
            symm_axes = symm.geom._refine(
                symm.geom._make_axes(c5_axes[0] + c5_axes[1], c5_axes[2] - c5_axes[5])
            )

    # check if molecule has symmetry of point group
    if correct_symm:

        return charge_center, symm_axes

    else:

        raise PointGroupSymmetryError(
            "Molecule does not have supplied symmetry. Maybe try reducing symmetry "
            "tolerance."
        )


def ao_trafo_mat(
    cart_mat: np.ndarray,
    nao: int,
    ao_loc: np.ndarray,
    l_shell: List[int],
    nctr_shell: List[int],
) -> np.ndarray:
    """
    this function generates an AO transformation matrix for arbitrary transformations
    """
    # get euler angles
    alpha, beta, gamma = symm.Dmatrix.get_euler_angles(np.eye(3), cart_mat)

    # generate transformation matrices for spherical harmonics
    sph_mats = [
        symm.Dmatrix.Dmatrix(l, alpha, beta, gamma, reorder_p=True)
        for l in range(max(l_shell) + 1)
    ]

    # initialize transformation matrix
    trafo_ao = np.zeros((nao, nao), dtype=np.float64)

    # loop over shells
    for shell, l in enumerate(l_shell):

        # loop over contracted basis functions in shell
        for bf in range(nctr_shell[shell]):

            # get ao index range for contracted basis function
            ao_start = ao_loc[shell] + bf * sph_mats[l].shape[1]
            ao_stop = ao_start + sph_mats[l].shape[1]

            # insert transformation matrix
            trafo_ao[ao_start:ao_stop, ao_start:ao_stop] = sph_mats[l]

    return trafo_ao


def get_mo_trafos(
    symm_trafo_ovlp: np.ndarray, tot_len: int, symm_tol: float
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    this function generates a list of symmetry-invariant and symmetry-equivalent MOs
    for a given overlap matrix
    """
    # intitialize list of symmetry equivalent orbitals
    symm_eqv_mos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    # get absolute of transformed orbital overlap
    symm_trafo_ovlp = np.abs(symm_trafo_ovlp)

    # initialize list that gathers important orbital contributions to transformed
    # orbital
    orb_contrib: List[Tuple[int, List[int]]] = []

    # calculate threshold up to which orbital contributions are considered
    thresh = symm_tol * np.max(symm_trafo_ovlp, axis=1)

    # get elements of with significant overlap
    signif_ovlp = symm_trafo_ovlp > thresh

    # loop over orbitals
    for orb in range(tot_len):

        # add empty list for current orbital
        orb_contrib.append((orb, signif_ovlp[orb].nonzero()[0].tolist()))

    # loop over orbitals until none are left
    while len(orb_contrib) > 0:

        # add current orbital to first tuple
        tup1 = [orb_contrib[0][0]]

        # add all orbitals that the current orbital transforms into to second tuple
        tup2 = set(orb_contrib[0][1])

        # delete current orbital
        del orb_contrib[0]

        # set orbital counter
        orb = 0

        # loop until all remaining orbitals are considered
        while orb < len(orb_contrib):

            # check if any orbital this orbital transforms into coincides with any
            # orbital in second tuple
            if not tup2.isdisjoint(orb_contrib[orb][1]):

                # add this orbital to first tuple
                tup1.append(orb_contrib[orb][0])

                # add orbital this orbital transforms into to second tuple
                tup2.update(orb_contrib[orb][1])

                # delete this orbital
                del orb_contrib[orb]

                # reset orbital counter
                orb = 0

            else:

                # increment orbital counter
                orb += 1

        # check if every set of orbitals transforms into another set of orbitals of the
        # same size
        if len(tup1) == len(tup2):

            # add set of orbitals
            symm_eqv_mos.append((tuple(tup1), tuple(tup2)))

        else:

            raise RuntimeError(
                "An error occured when trying to detect orbital symmetry."
            )

    return symm_eqv_mos


def get_symm_inv_blocks(
    all_symm_trafo_ovlp: np.ndarray, thresh_perc: float
) -> Tuple[List[List[int]], np.ndarray]:
    """
    this function finds blocks that are invariant with respect to all symmetry
    operations
    """
    # set threshold for sparse array
    thresh = thresh_perc * np.max(all_symm_trafo_ovlp, axis=0)

    # copy array
    thresh_trafo_ovlp = all_symm_trafo_ovlp.copy()

    # set elements below threshold to zero
    thresh_trafo_ovlp[thresh_trafo_ovlp < thresh] = 0.0

    # create sparse array
    sparse_trafo_ovlp = csr_array(thresh_trafo_ovlp)

    # determine optimal ordering of orbitals
    reorder = sc.sparse.csgraph.reverse_cuthill_mckee(sparse_trafo_ovlp)

    # reorder array
    thresh_trafo_ovlp = thresh_trafo_ovlp[reorder.reshape(-1, 1), reorder]

    # initialize list for mo blocks that are approximately invariant with respect to
    # all symmetry operations and add first block
    symm_inv_blocks: List[List[int]] = [[0]]

    # initialize row counter
    start = 1

    # perform until all orbitals are considered
    while True:

        # loop over mos
        for mo in range(start, all_symm_trafo_ovlp.shape[0]):

            # check if mos overlaps with last block
            if thresh_trafo_ovlp[mo, symm_inv_blocks[-1]].any():

                # add mo
                symm_inv_blocks[-1].append(mo)

            else:

                # create new block
                symm_inv_blocks.append([mo])

                # start at next mo
                start = mo + 1

                # block is finished
                break

        else:

            # all orbitals finished
            break

    return symm_inv_blocks, reorder


def get_symm_unique_mos(
    orbsym: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]], norb: int
) -> np.ndarray:
    """
    generate list of symmetry-unique MOs
    """
    # initialize list of symmetry-equivalent orbitals for every orbital
    all_symm_eqv_mos: List[Set[int]] = [set() for _ in range(norb)]

    # loop over symmetry operation
    for op_eqv_mos in orbsym:

        # loop over orbital tuple
        for tup in op_eqv_mos:

            # check if single orbital transforms into other single orbital
            if len(tup[0]) == 1:

                # add to set of symmetry-equivalent orbitals
                all_symm_eqv_mos[tup[0][0]].add(tup[1][0])

    # inititialze list of unique combinations of symmetry-equivalent orbitals
    symm_unique_mo_combs: List[Set[int]] = []

    # loop over orbitals until none are left
    while len(all_symm_eqv_mos) > 0:

        # add current orbital to first tuple
        symm_unique_mo_combs.append(all_symm_eqv_mos[0])

        # delete current orbital
        del all_symm_eqv_mos[0]

        # set orbital counter
        orb = 0

        # loop until all remaining orbitals are considered
        while orb < len(all_symm_eqv_mos):

            # check if any orbital this orbital transforms into coincides with any
            # orbital in second tuple
            if not symm_unique_mo_combs[-1].isdisjoint(all_symm_eqv_mos[orb]):

                # add this orbital to first tuple
                symm_unique_mo_combs[-1].update(all_symm_eqv_mos[orb])

                # delete this orbital
                del all_symm_eqv_mos[orb]

                # reset orbital counter
                orb = 0

            else:

                # increment orbital counter
                orb += 1

    # pick one orbital from every set
    symm_unique_mos = np.array([list(tup)[0] for tup in symm_unique_mo_combs])

    return symm_unique_mos


def _cubic_coords() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    this function defines the coordinates of specific points within a cube
    """
    sqrt2d2 = sqrt(2) / 2
    sqrt3d3 = sqrt(3) / 3

    corners = [
        np.array([sqrt3d3, sqrt3d3, sqrt3d3]),
        np.array([-sqrt3d3, -sqrt3d3, sqrt3d3]),
        np.array([sqrt3d3, -sqrt3d3, -sqrt3d3]),
        np.array([-sqrt3d3, sqrt3d3, -sqrt3d3]),
    ]

    edges = [
        np.array([sqrt2d2, sqrt2d2, 0.0]),
        np.array([sqrt2d2, -sqrt2d2, 0.0]),
        np.array([sqrt2d2, 0.0, sqrt2d2]),
        np.array([sqrt2d2, 0.0, -sqrt2d2]),
        np.array([0.0, sqrt2d2, sqrt2d2]),
        np.array([0.0, sqrt2d2, -sqrt2d2]),
    ]

    surfaces = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    return corners, edges, surfaces


def _icosahedric_coords() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    """
    this function defines the coordinates of specific points within an icosahedron
    """
    rec_gr = 2 / (1 + sqrt(5))
    corners = [
        np.array([rec_gr, 1.0, 0.0]),
        np.array([-rec_gr, 1.0, 0.0]),
        np.array([0.0, rec_gr, 1.0]),
        np.array([0.0, -rec_gr, 1.0]),
        np.array([1.0, 0.0, rec_gr]),
        np.array([1.0, 0.0, -rec_gr]),
    ]

    onepgrd2 = (1 + rec_gr) / 2
    onemgrd2 = (1 - rec_gr) / 2
    edges = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([rec_gr / 2, onepgrd2, 1 / 2]),
        np.array([rec_gr / 2, onemgrd2, 1 / 2]),
        np.array([-rec_gr / 2, onepgrd2, 1 / 2]),
        np.array([-rec_gr / 2, onemgrd2, 1 / 2]),
        np.array([onepgrd2, 1 / 2, rec_gr / 2]),
        np.array([onemgrd2, 1 / 2, rec_gr / 2]),
        np.array([onepgrd2, 1 / 2, -rec_gr / 2]),
        np.array([onemgrd2, 1 / 2, -rec_gr / 2]),
        np.array([1 / 2, rec_gr / 2, onepgrd2]),
        np.array([1 / 2, rec_gr / 2, onemgrd2]),
        np.array([1 / 2, -rec_gr / 2, onepgrd2]),
        np.array([1 / 2, -rec_gr / 2, onemgrd2]),
    ]

    onepgrd3 = (1 + rec_gr) / 3
    twopgrd3 = (2 + rec_gr) / 3
    surfaces = [
        np.array([onepgrd3, onepgrd3, onepgrd3]),
        np.array([-onepgrd3, onepgrd3, onepgrd3]),
        np.array([onepgrd3, -onepgrd3, onepgrd3]),
        np.array([onepgrd3, onepgrd3, -onepgrd3]),
        np.array([0.0, twopgrd3, 1 / 3]),
        np.array([0.0, -twopgrd3, 1 / 3]),
        np.array([twopgrd3, 1 / 3, 0.0]),
        np.array([-twopgrd3, 1 / 3, 0.0]),
        np.array([1 / 3, 0.0, twopgrd3]),
        np.array([1 / 3, 0.0, -twopgrd3]),
    ]

    return corners, edges, surfaces


def _ident_matrix(l_max: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function constructs a cartesian identity matrix and identity matrices for all
    spherical harmonics until l_max
    """
    return np.eye(3), [np.eye(2 * l + 1) for l in range(l_max + 1)]


def _inv_matrix(l_max: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function constructs a cartesian inversion matrix and inversion matrices for
    all spherical harmonics until l_max
    """
    return -np.eye(3), [(-1) ** l * np.eye(2 * l + 1) for l in range(l_max + 1)]


def _rot_matrix(
    axis: np.ndarray, angle: float, l_max: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function constructs a cartesian rotation matrix and rotation matrices using
    the Wigner D-matrices for all spherical harmonics until l_max
    """
    rot = _cart_rot_matrix(axis, angle)

    alpha, beta, gamma = symm.Dmatrix.get_euler_angles(np.eye(3), rot)

    Ds = [
        symm.Dmatrix.Dmatrix(l, alpha, beta, gamma, reorder_p=True)
        for l in range(l_max + 1)
    ]

    return rot, Ds


def _cart_rot_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    this function constructs a cartesian rotation matrix
    """
    axis = axis / np.linalg.norm(axis)

    sin_a = sin(angle)
    cos_a = cos(angle)

    rot = np.zeros((3, 3), dtype=np.float64)

    rot[0, 0] = cos_a + axis[0] ** 2 * (1 - cos_a)
    rot[0, 1] = axis[0] * axis[1] * (1 - cos_a) - axis[2] * sin_a
    rot[0, 2] = axis[0] * axis[2] * (1 - cos_a) + axis[1] * sin_a
    rot[1, 0] = axis[1] * axis[0] * (1 - cos_a) + axis[2] * sin_a
    rot[1, 1] = cos_a + axis[1] ** 2 * (1 - cos_a)
    rot[1, 2] = axis[1] * axis[2] * (1 - cos_a) - axis[0] * sin_a
    rot[2, 0] = axis[2] * axis[0] * (1 - cos_a) - axis[1] * sin_a
    rot[2, 1] = axis[2] * axis[1] * (1 - cos_a) + axis[0] * sin_a
    rot[2, 2] = cos_a + axis[2] ** 2 * (1 - cos_a)

    return rot


def _reflect_matrix(
    normal: np.ndarray, l_max: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function constructs a reflection matrix and reflection matrices for all
    spherical harmonics until l_max
    """
    cart_rot_mat, sph_rot_mats = _rot_matrix(normal, pi, l_max)
    cart_inv_mat, sph_inv_mats = _inv_matrix(l_max)

    return cart_rot_mat @ cart_inv_mat, [
        rot @ inv for rot, inv in zip(sph_rot_mats, sph_inv_mats)
    ]


def _rot_reflect_matrix(
    axis: np.ndarray, angle: float, l_max: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    constructs a cartesian rotation-reflection matrix and rotation-reflection matrices
    for all spherical harmonics until l_max
    """
    cart_rot_mat, sph_rot_mats = _rot_matrix(axis, angle, l_max)
    cart_reflect_mat, sph_reflect_mats = _reflect_matrix(axis, l_max)

    return cart_rot_mat @ cart_reflect_mat, [
        rot @ reflect for rot, reflect in zip(sph_rot_mats, sph_reflect_mats)
    ]


def _tet_c3(c3_1: np.ndarray, c3_2: np.ndarray) -> List[np.ndarray]:
    """
    this function generates C3 axes in the tetrahedral point groups from two starting
    axes
    """
    if np.dot(c3_2, c3_1) > 0:
        c3_2 = -c3_2

    c3_3 = _cart_rot_matrix(c3_1, np.pi * 2 / 3) @ c3_2
    c3_4 = _cart_rot_matrix(c3_1, 2 * np.pi * 2 / 3) @ c3_2

    return [c3_1, c3_2, c3_3, c3_4]


def _oct_c4(c4_1: np.ndarray, c4_2: np.ndarray) -> List[np.ndarray]:
    """
    this function generates C4 axes in the octahedral point group from two starting axes
    """
    c4_3 = np.cross(c4_1, c4_2)

    return [c4_1, c4_2, c4_3]


def _ico_c5(c5_1: np.ndarray, c5_2: np.ndarray) -> List[np.ndarray]:
    """
    this function generates C5 axes in the icosahedral point group from two starting
    axes
    """
    if np.dot(c5_2, c5_1) < 0:
        c5_2 = -c5_2

    c5_3 = _cart_rot_matrix(c5_1, np.pi * 6 / 5) @ c5_2
    c5_4 = _cart_rot_matrix(c5_1, 2 * np.pi * 6 / 5) @ c5_2
    c5_5 = _cart_rot_matrix(c5_1, 3 * np.pi * 6 / 5) @ c5_2
    c5_6 = _cart_rot_matrix(c5_1, 4 * np.pi * 6 / 5) @ c5_2

    return [c5_1, c5_2, c5_3, c5_4, c5_5, c5_6]
