"""
pre_assembly_check.py
=====================
Comprehensive inventory check for every file the CR matrix assembler needs.
Checks: existence, shape, dtype, value ranges, column names, internal consistency.

Run from repo root:
    python src/rates/pre_assembly_check.py

All PASS -> safe to proceed to assemble_cr_matrix.py
Any FAIL -> fix before proceeding
"""

import numpy as np
import pandas as pd
import os
import sys

IH_eV  = 13.6058
# Hydrogen ionisation energy — two physically distinct values
IH_SPECTROSCOPIC = 13.598434599702   # eV  True I_H (finite proton mass, NIST)
                                      # Used for: A coefficients, RR, 3BR, Saha checks
IH_RYDBERG       = 13.605693122990   # eV  R_inf * hc (infinite nuclear mass)
                                      # Used for: CCC thresholds, TICS, V&S, K_ion

PATHS = {
    'K_exc_full':     'data/processed/collisions/K_exc_full/K_exc_full.npy',
    'K_deexc_full':   'data/processed/collisions/K_exc_full/K_deexc_full.npy',
    'K_exc_sparse':   'data/processed/collisions/K_exc_full/K_exc_sparse.npy',
    'K_deexc_sparse': 'data/processed/collisions/K_exc_full/K_deexc_sparse.npy',
    'Te_grid_K':      'data/processed/collisions/K_exc_full/Te_grid_K.npy',
    'K_exc_meta':     'data/processed/collisions/K_exc_full/K_exc_meta.csv',
    'state_index':    'data/processed/collisions/K_exc_full/state_index.csv',
    'K_ion_final':    'data/processed/collisions/tics/K_ion_final.npy',
    'K_ion_meta':     'data/processed/collisions/tics/K_ion_final_meta.csv',
    'Te_grid_ion':    'data/processed/collisions/tics/Te_grid_ion.npy',
    'alpha_RR_res':   'data/processed/recombination/alpha_RR_resolved.npy',
    'alpha_RR_bund':  'data/processed/recombination/alpha_RR_bundled.npy',
    'alpha_3BR_res':  'data/processed/recombination/alpha_3BR_resolved.npy',
    'alpha_3BR_bund': 'data/processed/recombination/alpha_3BR_bundled.npy',
    'Te_grid_recomb': 'data/processed/recombination/Te_grid_recomb.npy',
    'recomb_meta':    'data/processed/recombination/recombination_meta.csv',
    'A_resolved':     'data/processed/Radiative/A_resolved.npy',
    'A_bund_res':     'data/processed/Radiative/A_bund_res.npy',
    'A_bund_bund':    'data/processed/Radiative/A_bund_bund.npy',
    'gamma_resolved': 'data/processed/Radiative/gamma_resolved.npy',
    'gamma_bundled':  'data/processed/Radiative/gamma_bundled.npy',
}

SHAPES = {
    'K_exc_full':    (43, 43, 50),
    'K_deexc_full':  (43, 43, 50),
    'K_exc_sparse':  (819, 50),
    'K_deexc_sparse':(819, 50),
    'Te_grid_K':     (50,),
    'K_ion_final':   (43, 50),
    'Te_grid_ion':   (50,),
    'alpha_RR_res':  (36, 50),
    'alpha_RR_bund': (7,  50),
    'alpha_3BR_res': (36, 50),
    'alpha_3BR_bund':(7,  50),
    'Te_grid_recomb':(50,),
    'A_resolved':    (36, 36),
    'A_bund_res':    (36, 7),
    'A_bund_bund':   (7,  7),
    'gamma_resolved':(36,),
    'gamma_bundled': (7,),
}

CSV_COLS = {
    'K_exc_meta':  ['sparse_idx','i','j','label_i','label_j','source','g_i','g_j','E_pn_eV'],
    'state_index': ['idx','label','n','l','bundled','g','I_eV'],
    'K_ion_meta': ['state_idx','n','l','label','I_n_eV','g_nl','source','accuracy'],
    'recomb_meta': ['state_idx','n','l','label','type','I_n_eV','g_nl','RR_source','3BR_source'],
}

passes = []
fails  = []

def chk(name, condition, detail=''):
    if condition:
        passes.append(name)
        print(f"  OK   {name}  {detail}")
    else:
        fails.append(name)
        print(f"  FAIL {name}  {detail}")
    return condition

# ── Load all files ────────────────────────────────────────────────────────────
print("="*70)
print("BLOCK 1 -- FILE EXISTENCE AND SHAPE")
print("="*70)

data = {}
for key, path in PATHS.items():
    exists = os.path.exists(path)
    chk(f"exists:{key}", exists, path)
    if not exists:
        continue
    if path.endswith('.npy'):
        arr = np.load(path)
        data[key] = arr
        if key in SHAPES:
            chk(f"shape:{key}", arr.shape == SHAPES[key],
                f"got={arr.shape} expected={SHAPES[key]}")
        chk(f"dtype:{key}", arr.dtype in [np.float64, np.float32],
            f"dtype={arr.dtype}")
    else:
        df = pd.read_csv(path)
        data[key] = df
        if key in CSV_COLS:
            missing = [c for c in CSV_COLS[key] if c not in df.columns]
            chk(f"cols:{key}", len(missing)==0,
                f"missing={missing}" if missing else f"all {len(CSV_COLS[key])} cols present")
        chk(f"nonempty:{key}", len(df)>0, f"{len(df)} rows")

# ── Block 2: Te grid consistency ──────────────────────────────────────────────
print()
print("="*70)
print("BLOCK 2 -- Te GRID CONSISTENCY")
print("="*70)

te_ref = data.get('Te_grid_K')
if te_ref is not None:
    chk("Te_grid length=50", len(te_ref)==50, f"len={len(te_ref)}")
    chk("Te_grid range 1..10 eV",
        abs(te_ref[0]-1.0)<0.001 and abs(te_ref[-1]-10.0)<0.001,
        f"[{te_ref[0]:.4f}, {te_ref[-1]:.4f}]")
    chk("Te_grid log-spaced",
        np.allclose(np.diff(np.log(te_ref)), np.diff(np.log(te_ref))[0], rtol=1e-6),
        "uniform log spacing")
    for k in ['Te_grid_ion', 'Te_grid_recomb']:
        if k in data:
            chk(f"Te_grid {k}==Te_grid_K",
                np.allclose(data[k], te_ref),
                f"max_diff={np.max(np.abs(data[k]-te_ref)):.2e}")

# ── Block 3: K_exc value checks ───────────────────────────────────────────────
print()
print("="*70)
print("BLOCK 3 -- K_exc VALUES")
print("="*70)

if 'K_exc_full' in data and 'K_exc_sparse' in data:
    Ke  = data['K_exc_full']
    Kes = data['K_exc_sparse']
    Te  = data['Te_grid_K']
    ti3 = int(np.argmin(np.abs(Te - 3.0)))

    chk("K_exc_full no negatives", bool((Ke >= 0).all()), f"neg={(Ke<0).sum()}")
    chk("K_exc_full no NaN",       not bool(np.isnan(Ke).any()))
    chk("K_exc_sparse no negatives", bool((Kes >= 0).all()))
    chk("K_exc_sparse no NaN",       not bool(np.isnan(Kes).any()))

    lower_nz = int(np.sum(np.tril(Ke[:,:,ti3], k=-1) != 0))
    chk("K_exc_full upper triangular only", lower_nz==0, f"lower_nonzero={lower_nz}")

    # Anchors
    k_1s2p_1eV  = float(Ke[0, 2, 0])
    k_1s2p_3eV  = float(Ke[0, 2, ti3])
    k_1s2p_10eV = float(Ke[0, 2, -1])
    chk("K_exc(1S->2P, Te=1eV)",  5e-13 < k_1s2p_1eV  < 5e-12, f"{k_1s2p_1eV:.4e} (expect ~7e-13)")
    chk("K_exc(1S->2P, Te=3eV)",  1e-10 < k_1s2p_3eV  < 1e-9,  f"{k_1s2p_3eV:.4e} (expect ~5e-10)")
    chk("K_exc(1S->2P, Te=10eV)", 1e-9  < k_1s2p_10eV < 1e-7,  f"{k_1s2p_10eV:.4e} (expect ~8e-9)")
    chk("K_exc(1S->2P) monotone with Te",
        bool(np.all(np.diff(Ke[0,2,:]) >= 0)), "monotone increasing")

    k_n9n10 = float(Ke[36, 37, ti3])
    chk("K_exc(n9->n10, Te=3eV) magnitude", 5e-5 < k_n9n10 < 5e-4,
        f"{k_n9n10:.4e} (expect ~1.6e-4)")

    # Sparse/dense consistency
    if 'K_exc_meta' in data:
        meta = data['K_exc_meta']
        max_err = 0.0
        for k in range(min(20, len(meta))):
            i, j = int(meta.iloc[k]['i']), int(meta.iloc[k]['j'])
            diff = float(np.max(np.abs(Kes[k,:] - Ke[i,j,:])))
            max_err = max(max_err, diff)
        chk("K_exc sparse==dense (first 20)", max_err < 1e-30, f"max_diff={max_err:.2e}")

# ── Block 4: K_deexc detailed balance ─────────────────────────────────────────
print()
print("="*70)
print("BLOCK 4 -- K_deexc DETAILED BALANCE")
print("="*70)

if all(k in data for k in ['K_exc_full','K_deexc_full','K_exc_meta','Te_grid_K']):
    Ke   = data['K_exc_full']
    Kd   = data['K_deexc_full']
    meta = data['K_exc_meta']
    Te   = data['Te_grid_K']
    ti3  = int(np.argmin(np.abs(Te - 3.0)))

    upper_nz = int(np.sum(np.triu(Kd[:,:,ti3], k=1) != 0))
    chk("K_deexc_full lower triangular only", upper_nz==0, f"upper_nonzero={upper_nz}")

    db_errors = []
    for _, row in meta.iloc[:50].iterrows():
        i, j = int(row['i']), int(row['j'])
        gi, gj = float(row['g_i']), float(row['g_j'])
        E_pn = float(row['E_pn_eV'])
        for ti in [0, ti3, -1]:
            ke = float(Ke[i, j, ti])
            kd = float(Kd[j, i, ti])
            if ke > 1e-50:
                exp = ke * (gi/gj) * np.exp(E_pn/Te[ti])
                if exp > 0:
                    db_errors.append(abs(kd/exp - 1)*100)
    max_db = max(db_errors) if db_errors else 0
    chk("Detailed balance (first 50 pairs, Te=1,3,10eV)",
        max_db < 0.01, f"max_err={max_db:.2e}%")

    asym = 0
    for i in range(43):
        for j in range(i+1, 43):
            if (Ke[i,j,ti3]>0) != (Kd[j,i,ti3]>0):
                asym += 1
    chk("Exc/deexc symmetry (nonzero pattern)", asym==0, f"asymmetric_pairs={asym}")

# ── Block 5: K_ion ────────────────────────────────────────────────────────────
print()
print("="*70)
print("BLOCK 5 -- K_ion VALUES")
print("="*70)

if 'K_ion_final' in data:
    Ki = data['K_ion_final']
    Te = data.get('Te_grid_ion', data.get('Te_grid_K'))
    chk("K_ion_final no negatives", bool((Ki>=0).all()), f"neg={(Ki<0).sum()}")
    chk("K_ion_final no NaN",       not bool(np.isnan(Ki).any()))
    chk("K_ion(1S) monotone with Te", bool(np.all(np.diff(Ki[0,:]) >= 0)), "monotone")
    if Ki.shape[0] > 36:
        chk("K_ion(n9) > K_ion(1S) at Te=10eV",
            bool(Ki[36,-1] > Ki[0,-1]),
            f"K_ion(n9)={Ki[36,-1]:.3e}  K_ion(1S)={Ki[0,-1]:.3e}")
    nonzero = Ki[Ki > 0]
    if len(nonzero):
        chk("K_ion range (1e-25..1e-5 cm3/s)",
            bool(nonzero.min() > 1e-25 and nonzero.max() < 1e-3,),
            f"min={nonzero.min():.2e}  max={nonzero.max():.2e}")
    if 'K_ion_meta' in data:
        chk("K_ion_meta rows=43", len(data['K_ion_meta'])==43)

# ── Block 6: Recombination ────────────────────────────────────────────────────
print()
print("="*70)
print("BLOCK 6 -- RECOMBINATION VALUES")
print("="*70)

if all(k in data for k in ['alpha_RR_res','alpha_RR_bund',
                             'alpha_3BR_res','alpha_3BR_bund']):
    RR_r = data['alpha_RR_res']
    RR_b = data['alpha_RR_bund']
    BR_r = data['alpha_3BR_res']
    BR_b = data['alpha_3BR_bund']
    Te   = data.get('Te_grid_recomb', data.get('Te_grid_K'))

    for name, arr in [('RR_res',RR_r),('RR_bund',RR_b),
                      ('3BR_res',BR_r),('3BR_bund',BR_b)]:
        chk(f"alpha_{name} no negatives", bool((arr>=0).all()))
        chk(f"alpha_{name} no NaN", not bool(np.isnan(arr).any()))

    rr_1S_1eV = float(RR_r[0, 0])
    chk("alpha_RR(1S, Te=1eV) ~ Johnson 1972 Eq.7 (1.4595e-13 cm3/s)",
        1.40e-13 < rr_1S_1eV < 1.52e-13, f"{rr_1S_1eV:.4e}  ref=1.4595e-13")

    # 3BR detailed balance vs K_ion
    if 'K_ion_final' in data:
        ti3_r = int(np.argmin(np.abs(Te - 3.0)))
        Te3   = float(Te[ti3_r])
        h_SI  = 6.62607e-34; me_SI = 9.10938e-31; eV_J = 1.60218e-19
        lam3  = (h_SI**2/(2*np.pi*me_SI*(Te3*eV_J)))**1.5 * 1e6
        g_2P  = 6; I_2 = IH_eV/4
        factor = (g_2P/2)*lam3*np.exp(I_2/Te3)
        Ki_2P  = float(data['K_ion_final'][2, ti3_r])
        exp_3BR = Ki_2P * factor
        act_3BR = float(BR_r[2, ti3_r])
        err_db  = abs(act_3BR/exp_3BR - 1)*100 if exp_3BR>0 else 999
        chk("alpha_3BR(2P,Te=3) = K_ion*Saha_factor",
            err_db < 0.1, f"err={err_db:.3f}%  actual={act_3BR:.4e}  expected={exp_3BR:.4e}")

    # l-distribution: alpha_RR(n,l) proportional to (2l+1)/n^2
    for n_test in [2, 5, 8]:
        start = sum(range(1, n_test))
        rates = RR_r[start:start+n_test, 0]
        fracs = rates / rates.sum() if rates.sum() > 0 else rates
        exp_fracs = np.array([(2*l+1)/n_test**2 for l in range(n_test)])
        exp_fracs /= exp_fracs.sum()
        err = float(np.max(np.abs(fracs - exp_fracs))) * 100
        chk(f"RR l-distribution n={n_test}", err < 0.01, f"max_err={err:.3f}%")

# ── Block 7: A coefficients ───────────────────────────────────────────────────
print()
print("="*70)
print("BLOCK 7 -- A COEFFICIENT VALUES")
print("="*70)

if all(k in data for k in ['A_resolved','A_bund_res','A_bund_bund',
                             'gamma_resolved','gamma_bundled']):
    Ar  = data['A_resolved']
    Abr = data['A_bund_res']
    Abb = data['A_bund_bund']
    gr  = data['gamma_resolved']
    gb  = data['gamma_bundled']

    chk("A_resolved no negatives",  bool((Ar>=0).all()))
    chk("A_bund_res no negatives",  bool((Abr>=0).all()))
    chk("A_bund_bund no negatives", bool((Abb>=0).all()))
    chk("gamma_resolved no negatives", bool((gr>=0).all()))
    chk("gamma_bundled no negatives",  bool((gb>=0).all()))

    chk("A_resolved diagonal=0", bool(np.all(np.diag(Ar)==0)),
        f"diag_nonzero={int(np.sum(np.diag(Ar)!=0))}")

    # Column sum check: A[i,j] = rate j->i, so col j sums to gamma[j]
    col_err = float(np.max(np.abs(Ar.sum(axis=0)[:10] - gr[:10]) / (gr[:10]+1e-30))) * 100
    chk("A_resolved col_sum == gamma_resolved (first 10)",
        col_err < 0.1, f"max_err={col_err:.3f}%")

    # NIST anchor: A(2P->1S) = 6.265e8 s^-1
    a_2p_1s = float(Ar[0, 2])
    chk("A(2P->1S) ~ NIST 6.265e8 s^-1",
        5e8 < a_2p_1s < 7e8, f"{a_2p_1s:.4e}  NIST=6.265e8")

    # Ground state: gamma(1S) = 0
    chk("gamma(1S) = 0 (no lower state)", gr[0] == 0.0, f"{gr[0]:.4e}")

    # 2P: only decay channel is 2P->1S
    chk("gamma(2P) == A(2P->1S)", abs(gr[2] - a_2p_1s)/(a_2p_1s+1e-30) < 0.001,
        f"gamma={gr[2]:.4e}  A={a_2p_1s:.4e}")

    # gamma_bundled = col_sum(A_bund_res + A_bund_bund)
    gb_calc = Abr.sum(axis=0) + Abb.sum(axis=0)
    err_gb = float(np.max(np.abs(gb_calc - gb) / (gb+1e-30))) * 100
    chk("gamma_bundled == col_sum(A_bund_res + A_bund_bund)",
        err_gb < 0.1, f"max_err={err_gb:.3f}%")

# ── Block 8: state_index ──────────────────────────────────────────────────────
print()
print("="*70)
print("BLOCK 8 -- STATE INDEX")
print("="*70)

if 'state_index' in data:
    si = data['state_index']
    chk("state_index 43 rows", len(si)==43, f"{len(si)}")
    chk("state_index sequential idx", list(si['idx'])==list(range(43)))
    chk("first 36 states resolved (l>=0)", bool(all(si.iloc[:36]['l'] >= 0)))
    chk("last 7 states bundled (l==-1)",   bool(all(si.iloc[36:]['l'] == -1)))
    bundled_n = list(si.iloc[36:]['n'])
    chk("bundled n values 9..15", bundled_n==list(range(9,16)))
    max_I = float(max(abs(row['I_eV'] - IH_eV/row['n']**2) for _,row in si.iterrows()))
    chk("I_eV == IH/n^2 for all states", max_I < 1e-4, f"max_err={max_I:.2e}")
    g_ok = all(
        int(row['g']) == (2*(2*int(row['l'])+1) if int(row['l'])>=0 else 2*int(row['n'])**2)
        for _,row in si.iterrows()
    )
    chk("g == 2(2l+1) or 2n^2 for all states", g_ok)

# ── Block 9: Cross-file consistency ───────────────────────────────────────────
print()
print("="*70)
print("BLOCK 9 -- CROSS-FILE CONSISTENCY")
print("="*70)

if 'K_exc_full' in data and 'state_index' in data:
    chk("K_exc_full N_states == len(state_index)",
        data['K_exc_full'].shape[0] == len(data['state_index']),
        f"{data['K_exc_full'].shape[0]}=={len(data['state_index'])}")

if 'K_ion_final' in data:
    chk("K_ion_final covers all 43 states", data['K_ion_final'].shape[0]==43)

if 'alpha_RR_res' in data and 'alpha_RR_bund' in data:
    total = data['alpha_RR_res'].shape[0] + data['alpha_RR_bund'].shape[0]
    chk("RR resolved(36)+bundled(7)==43", total==43, f"{total}")

if 'K_exc_meta' in data:
    meta = data['K_exc_meta']
    chk("K_exc_meta i,j in [0,42]",
        bool(meta['i'].between(0,42).all() and meta['j'].between(0,42).all()))
    chk("K_exc_meta i < j (excitation direction)",
        bool((meta['i'] < meta['j']).all()),
        f"violations={(meta['i']>=meta['j']).sum()}")
    chk("K_exc_meta sources valid",
        bool(meta['source'].isin(['CCC','CCC_n9','CCC_n10','VS']).all()),
        f"unique={sorted(meta['source'].unique())}")
    src_counts = meta['source'].value_counts().to_dict()
    chk("CCC pairs count=546",    src_counts.get('CCC',0)==546,    f"{src_counts.get('CCC',0)}")
    chk("CCC_n9 pairs count=36",  src_counts.get('CCC_n9',0)==36,  f"{src_counts.get('CCC_n9',0)}")
    chk("CCC_n10 pairs count=36", src_counts.get('CCC_n10',0)==36, f"{src_counts.get('CCC_n10',0)}")
    chk("VS pairs count=201",     src_counts.get('VS',0)==201,     f"{src_counts.get('VS',0)}")

te_ref = data.get('Te_grid_K')
if te_ref is not None:
    for k in ['Te_grid_ion', 'Te_grid_recomb']:
        if k in data:
            diff = float(np.max(np.abs(data[k] - te_ref)))
            chk(f"Te_grid {k} == Te_grid_K", diff < 1e-12, f"max_diff={diff:.2e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("="*70)
print("FINAL SUMMARY")
print("="*70)
total = len(passes) + len(fails)
print(f"\n  PASSED: {len(passes)}/{total}")
print(f"  FAILED: {len(fails)}/{total}")
if fails:
    print("\n  Failed checks:")
    for f in fails:
        print(f"    FAIL  {f}")
    print()
    print("  Fix the above before running assemble_cr_matrix.py")
    sys.exit(1)
else:
    print()
    print("  ALL CHECKS PASS -- ready for assemble_cr_matrix.py")
    sys.exit(0)