from pyscf import gto, scf, lo
from symlo import symmetrize_mos

# create mol object
mol = gto.Mole()
mol.build(
    verbose=0,
    atom="""
    O
    H   1 0.958
    H   1 0.958 2 104.45
    """,
    basis="cc-pvdz",
    symmetry=True,
)

# hf calculation
hf = scf.RHF(mol).run(conv_tol=1e-10)

# copy mo coefficients
mo_coeff = hf.mo_coeff.copy()

# localize occupied orbitals
loc = lo.Boys(mol, mo_coeff=mo_coeff[:, : min(mol.nelec)]).set(conv_tol=1e-10)
mo_coeff[:, : min(mol.nelec)] = loc.kernel()

# localize virtual orbitals
loc = lo.Boys(mol, mo_coeff=mo_coeff[:, max(mol.nelec) :]).set(conv_tol=1e-10)
mo_coeff[:, max(mol.nelec) :] = loc.kernel()

# symmetrize occupied localized orbitals
orbsym_occ, mo_coeff[:, : min(mol.nelec)] = symmetrize_mos(
    mol,
    mo_coeff[:, : min(mol.nelec)],
    mol.topgroup,
    verbose=4,
    max_cycle=100,
    conv_tol=1e-13,
    inv_block_thresh=0.3,
    symm_eqv_thresh=0.3,
)

# symmetrize virtual localized orbitals
orbsym_virt, mo_coeff[:, max(mol.nelec) :] = symmetrize_mos(
    mol,
    mo_coeff[:, max(mol.nelec) :],
    mol.topgroup,
    verbose=4,
    max_cycle=100,
    conv_tol=1e-13,
    inv_block_thresh=0.3,
    symm_eqv_thresh=0.3,
)
