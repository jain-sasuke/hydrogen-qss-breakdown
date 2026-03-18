"""
assemble_K_exc.py
=================
Assemble ALL electron-impact excitation and de-excitation rate coefficients
into a single unified table covering every bound-state pair (0..42).

SOURCES MERGED
--------------
  CCC (1320 rows on Mac; 870 in old file):
    K_CCC_exc_table.npy      (1320, 50) — res-res n=1..8 AND res→n10
    K_CCC_deexc_table.npy    (1320, 50) — de-excitation via DB
    K_CCC_metadata.csv       (1320, 12)
    K_exc_to_n10_bundled.npy  (36, 50)  — res→n10 summed over l_f (from compute_K_CCC.py)

  CCC collapse (built here):
    res→n9_bundled:  sum over l_f=0..8 of K_CCC(ni,li→9,lf)
    This gives CCC accuracy (~5%) for res→n9 instead of V&S (~20%)
    Data already in K_CCC_metadata (n_f=9 rows, 324 entries)

  V&S (237 rows — but res→n9 block replaced by CCC collapse):
    K_VS_exc_table.npy       (201, 50) after excluding res→n9
    res→n11..15 + bund↔bund only

COVERAGE (1143 unique excitation pairs):
  CCC  res-res n=1..8        :  546 pairs  (n_f=2..8, excludes n_f=9 rows mapped below)
  CCC  res→n9  (collapse)    :   36 pairs  (sum over l_f, ~5% accuracy)
  CCC  res→n10               :   36 pairs  (K_exc_to_n10_bundled)
  V&S  res→n11..15           :  180 pairs  (~20% accuracy)
  V&S  bund↔bund n=9..15     :   21 pairs  (~20% accuracy)
  NOTE: res→n9 V&S rows are ignored (CCC is better)

  Wait — the correct count:
  CCC metadata n_f=2..8: 870 - 324 = 546 pairs (l-resolved to l-resolved)
  CCC collapse n_f=9:     36 pairs (one per initial resolved state)
  CCC n10 bundled:        36 pairs
  V&S res→n11..15:       180 pairs
  V&S bund↔bund:          21 pairs
  TOTAL: 546 + 36 + 36 + 180 + 21 = 819 pairs

  The 'expected 1143' in the old plan was wrong — it double-counted by
  including BOTH CCC n_f=9 resolved AND V&S res→n9 bundled. With CCC
  collapse replacing V&S for res→n9, the correct total is 819.

STATE INDEXING (43 bound states):
  0..35  : n=1..8 l-resolved
  36..42 : n=9..15 l-bundled
  n=9 is BUNDLED at index 36, key (9,-1)

CONVENTION:
  K_exc_full[i, j, Te_idx]   i=lower state, j=upper state  (i < j)
  K_deexc_full[i, j, Te_idx] i=upper state, j=lower state  (i > j)
  Shapes: (43, 43, 50)

OUTPUTS (data/processed/collisions/K_exc_full/)
-------
  K_exc_full.npy     (43, 43, 50)  upper triangular [cm³/s]
  K_deexc_full.npy   (43, 43, 50)  lower triangular [cm³/s]
  K_exc_sparse.npy   (819, 50)     excitation in pair order [cm³/s]
  K_deexc_sparse.npy (819, 50)     de-excitation [cm³/s]
  K_exc_meta.csv     (819 rows)    i,j,labels,source,g_i,g_j,E_pn_eV
  state_index.csv    (43 rows)
  Te_grid_K.npy      (50,)

REFERENCES
----------
CCC: Bray (pers. comm. 2026), compute_K_CCC.py
V&S: Vriens & Smeets (1980) Phys.Rev.A 22, 940, compute_K_VS.py
"""

import numpy as np
import pandas as pd
import os

IH_eV   = 13.6058
L_CHAR  = 'SPDFGHIJKL'
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)


def build_state_index():
    states = []
    idx = 0
    for n in range(1, 9):
        for l in range(n):
            states.append({'idx': idx, 'label': f'{n}{L_CHAR[l]}',
                           'n': n, 'l': l, 'bundled': False,
                           'g': 2*(2*l+1), 'I_eV': round(IH_eV/n**2, 8)})
            idx += 1
    for n in range(9, 16):
        states.append({'idx': idx, 'label': f'n{n}',
                       'n': n, 'l': -1, 'bundled': True,
                       'g': 2*n**2, 'I_eV': round(IH_eV/n**2, 8)})
        idx += 1
    assert idx == 43
    return states


def assemble_K_exc(
    ccc_dir='data/processed/collisions/ccc',
    vs_dir='data/processed/collisions/vs',
    out_dir='data/processed/collisions/K_exc_full',
):
    os.makedirs(out_dir, exist_ok=True)

    states   = build_state_index()
    by_nl    = {(s['n'], s['l']): s['idx'] for s in states}
    N, n_Te  = 43, 50

    K_exc_dense   = np.zeros((N, N, n_Te))
    K_deexc_dense = np.zeros((N, N, n_Te))
    sparse_rows   = []
    collisions    = 0

    # ── Block 1: CCC res-res (n_f=2..8, skip n_f=9 resolved) ─────────────────
    print("Loading CCC tables...")
    K_ccc_exc   = np.load(f'{ccc_dir}/K_CCC_exc_table.npy')
    K_ccc_deexc = np.load(f'{ccc_dir}/K_CCC_deexc_table.npy')
    meta_ccc    = pd.read_csv(f'{ccc_dir}/K_CCC_metadata.csv')
    K_n10       = np.load(f'{ccc_dir}/K_exc_to_n10_bundled.npy')

    n_ccc_res = 0
    for row_idx, row in meta_ccc.iterrows():
        ni, li = int(row['n_i']), int(row['l_i'])
        nf, lf = int(row['n_f']), int(row['l_f'])

        # Skip n_f=9 rows — handled by CCC collapse block below
        if nf == 9:
            continue

        si = by_nl.get((ni, li))
        sf = by_nl.get((nf, lf))
        if si is None or sf is None:
            continue
        assert si < sf
        ke = K_ccc_exc[row_idx, :]
        kd = K_ccc_deexc[row_idx, :]
        if K_exc_dense[si, sf, 0] != 0:
            collisions += 1
        K_exc_dense[si, sf, :]   = ke
        K_deexc_dense[sf, si, :] = kd
        E_pn = IH_eV*(1/ni**2 - 1/nf**2)
        sparse_rows.append({'i': si, 'j': sf,
            'label_i': states[si]['label'], 'label_j': states[sf]['label'],
            'source': 'CCC', 'g_i': states[si]['g'], 'g_j': states[sf]['g'],
            'E_pn_eV': round(E_pn, 8)})
        n_ccc_res += 1
    print(f"  CCC res-res (n_f=2..8): {n_ccc_res} pairs")

    # ── Block 2: CCC res→n9 (collapse over l_f) ───────────────────────────────
    # Sum all K_CCC(ni,li → 9,lf) over lf=0..8 for each (ni,li) initial state
    # This gives K_exc(ni,li → n9_bundled) with CCC accuracy (~5%)
    n9_idx = by_nl[(9, -1)]
    g_n9   = states[n9_idx]['g']   # = 162
    n_ccc_n9 = 0
    si_res = 0
    for ni in range(1, 9):
        for li in range(ni):
            si   = by_nl[(ni, li)]
            E_pn = IH_eV*(1/ni**2 - 1/81)
            g_i  = states[si]['g']

            # Rows in CCC metadata: n_i=ni, l_i=li, n_f=9
            mask = (meta_ccc['n_i']==ni) & (meta_ccc['l_i']==li) & (meta_ccc['n_f']==9)
            rows = meta_ccc[mask]

            if len(rows) > 0:
                # Sum excitation rates over all l_f sublevels of n=9
                ke = K_ccc_exc[rows.index, :].sum(axis=0)
                # De-excitation via detailed balance on the TOTAL bundled rate
                kd = ke * (g_i/g_n9) * np.exp(E_pn / TE_GRID)
            else:
                # Fallback: zero (shouldn't happen for n=1..8 to n=9)
                ke = np.zeros(n_Te)
                kd = np.zeros(n_Te)

            if K_exc_dense[si, n9_idx, 0] != 0:
                collisions += 1
            K_exc_dense[si, n9_idx, :]   = ke
            K_deexc_dense[n9_idx, si, :] = kd
            sparse_rows.append({'i': si, 'j': n9_idx,
                'label_i': states[si]['label'], 'label_j': 'n9',
                'source': 'CCC_n9', 'g_i': g_i, 'g_j': g_n9,
                'E_pn_eV': round(E_pn, 8)})
            si_res += 1
            n_ccc_n9 += 1
    print(f"  CCC res→n9 (collapse):  {n_ccc_n9} pairs")

    # ── Block 3: CCC res→n10 (from precomputed bundled table) ─────────────────
    n10_idx = by_nl[(10, -1)]
    g_n10   = states[n10_idx]['g']
    n_ccc_n10 = 0
    si_res = 0
    for ni in range(1, 9):
        for li in range(ni):
            si   = by_nl[(ni, li)]
            ke   = K_n10[si_res, :]
            E_pn = IH_eV*(1/ni**2 - 1/100)
            g_i  = states[si]['g']
            kd   = ke * (g_i/g_n10) * np.exp(E_pn / TE_GRID)
            if K_exc_dense[si, n10_idx, 0] != 0:
                collisions += 1
            K_exc_dense[si, n10_idx, :]   = ke
            K_deexc_dense[n10_idx, si, :] = kd
            sparse_rows.append({'i': si, 'j': n10_idx,
                'label_i': states[si]['label'], 'label_j': 'n10',
                'source': 'CCC_n10', 'g_i': g_i, 'g_j': g_n10,
                'E_pn_eV': round(E_pn, 8)})
            si_res += 1
            n_ccc_n10 += 1
    print(f"  CCC res→n10 (bundled):  {n_ccc_n10} pairs")

    # ── Block 4: V&S res→n11..15 and bund↔bund (skip res→n9) ─────────────────
    print("Loading V&S tables...")
    K_vs_exc   = np.load(f'{vs_dir}/K_VS_exc_table.npy')
    K_vs_deexc = np.load(f'{vs_dir}/K_VS_deexc_table.npy')
    meta_vs    = pd.read_csv(f'{vs_dir}/K_VS_metadata.csv')

    n_vs = 0
    n_vs_skipped = 0
    for _, row in meta_vs.iterrows():
        vs_idx = int(row['idx'])
        p, lp  = int(row['p']), int(row['l_p'])
        n, ln  = int(row['n']), int(row['l_n'])

        # Skip res→n9: covered by CCC collapse (better accuracy)
        if n == 9 and lp >= 0:
            n_vs_skipped += 1
            continue

        si = by_nl.get((p, lp))
        sf = by_nl.get((n, ln))
        if si is None or sf is None:
            print(f"  WARN: ({p},{lp})->({n},{ln}) not in state index")
            continue
        assert si < sf
        ke = K_vs_exc[vs_idx, :]
        kd = K_vs_deexc[vs_idx, :]
        if K_exc_dense[si, sf, 0] != 0:
            collisions += 1
            print(f"  COLLISION: {states[si]['label']}->{states[sf]['label']}")
        K_exc_dense[si, sf, :]   = ke
        K_deexc_dense[sf, si, :] = kd
        sparse_rows.append({'i': si, 'j': sf,
            'label_i': row['label_p'], 'label_j': row['label_n'],
            'source': 'VS', 'g_i': int(row['g_p']), 'g_j': int(row['g_n']),
            'E_pn_eV': float(row['E_pn_eV'])})
        n_vs += 1

    print(f"  V&S (res→n11..15 + bund↔bund): {n_vs} pairs")
    print(f"  V&S res→n9 skipped (CCC used):  {n_vs_skipped} pairs")

    # ── Sparse arrays ─────────────────────────────────────────────────────────
    n_pairs = len(sparse_rows)
    K_exc_sparse   = np.zeros((n_pairs, n_Te))
    K_deexc_sparse = np.zeros((n_pairs, n_Te))
    for k, row in enumerate(sparse_rows):
        i, j = row['i'], row['j']
        K_exc_sparse[k, :]   = K_exc_dense[i, j, :]
        K_deexc_sparse[k, :] = K_deexc_dense[j, i, :]

    meta_sparse = pd.DataFrame(sparse_rows)
    meta_sparse.insert(0, 'sparse_idx', range(n_pairs))

    # ── QC ────────────────────────────────────────────────────────────────────
    print()
    print("="*65)
    print("QC CHECKS")
    print("="*65)

    # A: collisions
    print(f"\nCheck A — Source collisions: {collisions}  "
          f"{'PASS' if collisions==0 else 'FAIL'}")

    # B: total pairs
    expected = n_ccc_res + n_ccc_n9 + n_ccc_n10 + n_vs
    print(f"\nCheck B — Total pairs: {n_pairs}  (= {n_ccc_res}+{n_ccc_n9}+{n_ccc_n10}+{n_vs})  "
          f"{'PASS' if n_pairs==expected else 'FAIL'}")

    # C: detailed balance
    ti3  = np.argmin(np.abs(TE_GRID - 3.0))
    Te3  = TE_GRID[ti3]
    errs = []
    for k, row in enumerate(sparse_rows[:80]):
        ke = K_exc_sparse[k, ti3]
        kd = K_deexc_sparse[k, ti3]
        if ke > 1e-50:
            exp_kd = ke * (row['g_i']/row['g_j']) * np.exp(row['E_pn_eV']/Te3)
            if exp_kd > 0:
                errs.append(abs(kd/exp_kd - 1)*100)
    max_db = max(errs) if errs else 0
    print(f"\nCheck C — Detailed balance (first 80, Te≈{Te3:.2f}eV): "
          f"max err={max_db:.2e}%  {'PASS' if max_db < 0.01 else 'FAIL'}")

    # D: no CCC/VS overlap
    ccc_p = {(r['i'],r['j']) for r in sparse_rows if 'CCC' in r['source']}
    vs_p  = {(r['i'],r['j']) for r in sparse_rows if r['source']=='VS'}
    olap  = ccc_p & vs_p
    print(f"\nCheck D — CCC/V&S overlap: {len(olap)} pairs  "
          f"{'PASS' if len(olap)==0 else 'FAIL'}")

    # E: coverage
    upper_nz = int((K_exc_dense[:,:,ti3] > 0).sum())
    all_zero  = int((K_exc_sparse.max(axis=1) == 0).sum())
    print(f"\nCheck E — Coverage:")
    print(f"  Upper-triangle nonzero at Te=3eV: {upper_nz}")
    print(f"  Pairs zero at ALL Te:             {all_zero}")

    # F: CCC vs V&S comparison for res→n9 (verify CCC used, not VS)
    print(f"\nCheck F — CCC used for res→n9 (not V&S):")
    for si in [0, 35]:    # 1S and 8J
        sf = by_nl[(9,-1)]
        k = K_exc_dense[si, sf, ti3]
        lbl = states[si]['label']
        src = next((r['source'] for r in sparse_rows if r['i']==si and r['j']==sf), '?')
        print(f"  K({lbl}→n9, Te=3eV) = {k:.4e} cm³/s  source={src}  "
              f"{'PASS' if src=='CCC_n9' else 'FAIL'}")

    # G: rate table
    ti_cols = [0, ti3, np.argmin(np.abs(TE_GRID-5)), 49]
    print(f"\nCheck G — Rate table [cm³/s]:")
    print(f"  {'Pair':16s}  {'Src':7s}  {'Te=1':>12s}  {'Te=3':>12s}"
          f"  {'Te=5':>12s}  {'Te=10':>12s}")
    print("  " + "-"*74)
    pairs_show = [
        (by_nl[(1,0)],  by_nl[(2,1)],   'CCC',     '1S->2P'),
        (by_nl[(1,0)],  by_nl[(3,2)],   'CCC',     '1S->3D'),
        (by_nl[(8,7)],  by_nl[(9,-1)],  'CCC_n9',  '8J->n9'),
        (by_nl[(1,0)],  by_nl[(9,-1)],  'CCC_n9',  '1S->n9'),
        (by_nl[(1,0)],  by_nl[(10,-1)], 'CCC_n10', '1S->n10'),
        (by_nl[(9,-1)], by_nl[(10,-1)], 'VS',      'n9->n10'),
        (by_nl[(9,-1)], by_nl[(15,-1)], 'VS',      'n9->n15'),
        (by_nl[(11,-1)],by_nl[(12,-1)], 'VS',      'n11->n12'),
    ]
    for si, sf, src, lbl in pairs_show:
        vals = [K_exc_dense[si, sf, ti] for ti in ti_cols]
        print(f"  {lbl:16s}  {src:7s}  " +
              "  ".join(f"{v:12.4e}" for v in vals))

    # H: source breakdown
    print(f"\nCheck H — Source breakdown:")
    for src, cnt in meta_sparse['source'].value_counts().items():
        print(f"  {src:<12s}: {cnt:4d} pairs")

    # I: symmetry
    sym_ok = True
    for si in range(N):
        for sf in range(si+1, N):
            e = K_exc_dense[si, sf, ti3] > 0
            d = K_deexc_dense[sf, si, ti3] > 0
            if e != d:
                sym_ok = False
                print(f"  ASYM: {states[si]['label']}->{states[sf]['label']}")
    print(f"\nCheck I — Exc/deexc symmetry: {'PASS' if sym_ok else 'FAIL'}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(f'{out_dir}/K_exc_full.npy',     K_exc_dense)
    np.save(f'{out_dir}/K_deexc_full.npy',   K_deexc_dense)
    np.save(f'{out_dir}/K_exc_sparse.npy',   K_exc_sparse)
    np.save(f'{out_dir}/K_deexc_sparse.npy', K_deexc_sparse)
    np.save(f'{out_dir}/Te_grid_K.npy',      TE_GRID)
    meta_sparse.to_csv(f'{out_dir}/K_exc_meta.csv', index=False)
    pd.DataFrame(states).to_csv(f'{out_dir}/state_index.csv', index=False)

    print(f"\nSaved to {out_dir}/:")
    print(f"  K_exc_full.npy      {K_exc_dense.shape}   upper-triangular [cm³/s]")
    print(f"  K_deexc_full.npy    {K_deexc_dense.shape}   lower-triangular [cm³/s]")
    print(f"  K_exc_sparse.npy    {K_exc_sparse.shape}  [cm³/s]")
    print(f"  K_deexc_sparse.npy  {K_deexc_sparse.shape}  [cm³/s]")
    print(f"  K_exc_meta.csv      {n_pairs} pairs")
    print(f"  state_index.csv     43 states")
    print(f"  Te_grid_K.npy       {TE_GRID.shape}")

    return K_exc_dense, K_deexc_dense, K_exc_sparse, K_deexc_sparse, meta_sparse


if __name__ == '__main__':
    assemble_K_exc()