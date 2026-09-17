"""
CCC vs Anderson (2002 corrigendum) RMPS benchmark
=================================================

Validates the Bray CCC electron-impact excitation data against the
R-matrix-with-pseudostates (RMPS) effective collision strengths of

    H Anderson, C P Ballance, N R Badnell, H P Summers
    J. Phys. B: At. Mol. Opt. Phys. 35 (2002) 1613   [CORRIGENDUM]

which *replaces* table 2 of Anderson et al (2000) J. Phys. B 33 1255.

Why the corrigendum matters
---------------------------
Dere & Mason pointed out that the 2000 dipole effective collision strengths
do not approach the correct high-energy asymptotic form. The underlying
collision strengths were fine; the Maxwell averaging used a least-squares fit
to the high-energy tail, and RMPS collision strengths oscillate (pseudo-
thresholds), so the fit degraded beyond the last computed energy. The 2002
table recomputes Upsilon with linear interpolation of the reduced collision
strength between the highest finite energy and the infinite-energy limit
(Born limit for non-dipole). Anderson et al state the change is <~10% below
15 eV -- this script measures that instead of assuming it.

Provenance
----------
Upsilon values are PARSED FROM THE PDF at refs/, not retyped. The parsed
table is written out so every number in the comparison is traceable.

Anderson Eq. (3), excitation rate coefficient [cm^3/s]:

    q_exc = (2 sqrt(pi) alpha c a0^2 / omega_lower)
            * sqrt(I_H / kTe) * exp(-dE / kTe) * Upsilon

with omega_lower = (2S+1)(2L+1) = 2(2l+1) for hydrogen doublets.

Table convention: rows are "i j" with i = UPPER, j = LOWER (confirmed by the
A_ij column: row "3 1" carries A = 6.27e8 s^-1 = A(2p->1s)). This script
re-verifies that convention at runtime rather than trusting it.

Run:  python src/validation/anderson2002_benchmark.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Repo root autodiscovery ───────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

PDF_PATH  = os.path.join(ROOT, "refs",
                         "H_Anderson_2002_J._Phys._B__At._Mol._Opt._Phys._35_1613.pdf")
CCC_PATH  = os.path.join(ROOT, "data", "processed", "collisions", "ccc",
                         "ccc_crosssections.csv")
OUT_DIR   = os.path.join(ROOT, "data", "processed", "collisions")
FIG_DIR   = os.path.join(ROOT, "figures")

for p, what in [(PDF_PATH, "Anderson 2002 corrigendum PDF"),
                (CCC_PATH, "CCC cross-section table")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {what}: {p}")

# ── Physical constants (CODATA, cgs where noted) ──────────────────────────────
eV_to_J = 1.60218e-19
me      = 9.10938e-31          # [kg]
a0_m    = 5.29177e-11          # Bohr radius [m]
a0_cm   = 5.29177e-9           # Bohr radius [cm]
alpha   = 7.29735e-3
c_cgs   = 2.99792e10           # [cm/s]
IH_eV   = 13.6058              # Rydberg [eV]

C_AND = 2.0 * np.sqrt(np.pi) * alpha * c_cgs * a0_cm**2   # Anderson Eq.(3) prefactor

TE_AND = np.array([0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0])   # Anderson Te grid [eV]
THESIS_TE_IDX = [1, 2, 3, 4]            # Te = 1, 3, 5, 10 eV -- the thesis range
THESIS_TE     = TE_AND[THESIS_TE_IDX]

# Anderson table 1 level index -> (n, l)
IDX_TO_NL = {
    1: (1, 0),  2: (2, 0),  3: (2, 1),
    4: (3, 0),  5: (3, 1),  6: (3, 2),
    7: (4, 0),  8: (4, 1),  9: (4, 2), 10: (4, 3),
   11: (5, 0), 12: (5, 1), 13: (5, 2), 14: (5, 3), 15: (5, 4),
}
L_CHAR = ["S", "P", "D", "F", "G", "H"]

nl_label     = lambda n, l: f"{n}{L_CHAR[l]}"
stat_weight  = lambda l: 2 * (2 * l + 1)                 # (2S+1)(2L+1), doublets
threshold_eV = lambda n_lo, n_up: IH_eV * (1.0 / n_lo**2 - 1.0 / n_up**2)


# ── 1. Parse Upsilon table out of the 2002 corrigendum PDF ────────────────────
def parse_anderson2002(pdf_path):
    """Return DataFrame: i_upper, j_lower, A_ij, and Upsilon at the 8 Te points."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text = "\n".join(pg.extract_text() for pg in reader.pages)
    # PDF uses U+2212 MINUS SIGN and mantissa/exponent may be space-separated
    text = text.replace("−", "-").replace("‐", "-")

    num = re.compile(r"(\d\.\d{2})\s*([+-])\s*(\d{2})")
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{1,2})\s+(\d{1,2})\s+(?=\d\.\d{2})", line)
        if not m:
            continue
        i_up, j_lo = int(m.group(1)), int(m.group(2))
        if i_up not in IDX_TO_NL or j_lo not in IDX_TO_NL:
            continue
        vals = [float(f"{a}e{s}{e}") for a, s, e in num.findall(line)]
        if len(vals) != 9:            # A_ij + 8 temperatures
            raise ValueError(f"Parsed {len(vals)} numbers (expected 9) from: {line!r}")
        rows.append([i_up, j_lo, vals[0]] + vals[1:])

    df = pd.DataFrame(rows, columns=["i_upper", "j_lower", "A_ij"] +
                                    [f"ups_{t:g}eV" for t in TE_AND])

    # --- parse integrity checks: fail loudly, do not paper over ---
    n_expected = 85          # C(15,2)=105 pairs minus 20 same-n pairs
    if len(df) != n_expected:
        raise ValueError(f"Parsed {len(df)} rows from PDF, expected {n_expected}. "
                         "PDF text extraction has changed -- do not trust the numbers.")
    if df.duplicated(["i_upper", "j_lower"]).any():
        raise ValueError("Duplicate (i,j) rows parsed from PDF.")
    for i_up, j_lo in zip(df.i_upper, df.j_lower):
        if i_up <= j_lo:
            raise ValueError(f"Row ({i_up},{j_lo}): i must be the UPPER index.")
        if IDX_TO_NL[i_up][0] == IDX_TO_NL[j_lo][0]:
            raise ValueError(f"Row ({i_up},{j_lo}) is a same-n pair; table excludes these.")

    # --- convention check: A(2p->1s) must appear in row (3,1) ---
    a31 = df.loc[(df.i_upper == 3) & (df.j_lower == 1), "A_ij"].iloc[0]
    if not (6.0e8 < a31 < 6.5e8):
        raise ValueError(f"Row (3,1) A_ij = {a31:.3e}, expected ~6.27e8 s^-1 "
                         "= A(2p->1s). The i/j = upper/lower convention is NOT confirmed.")
    print(f"  convention check: row (3,1) A_ij = {a31:.3e} s^-1 "
          f"= A(2p->1s) [NIST 6.2649e8]  OK")
    return df


# ── 2. Rate-coefficient helpers ───────────────────────────────────────────────
def K_from_upsilon(upsilon, n_lo, l_lo, n_up, Te_eV):
    """Anderson Eq.(3): effective collision strength -> excitation rate [cm^3/s]."""
    return (C_AND / stat_weight(l_lo)
            * np.sqrt(IH_eV / Te_eV)
            * np.exp(-threshold_eV(n_lo, n_up) / Te_eV)
            * upsilon)


def K_maxwell(sig_a0sq, E_eV, Te_eV):
    """Maxwellian average of sigma(E) -> rate coefficient [cm^3/s]. SI internally."""
    sig_m2 = sig_a0sq * a0_m**2
    E_J    = E_eV * eV_to_J
    kTe_J  = Te_eV * eV_to_J
    prefac = np.sqrt(8.0 / np.pi / me) * kTe_J**-1.5
    integ  = sig_m2 * E_J * np.exp(-E_eV / Te_eV)
    return prefac * np.trapezoid(integ, E_J) * 1e6      # m^3/s -> cm^3/s


# ── 3. Anderson (2000) table, as hardcoded in anderson_benchmark_qc.py ────────
#     Imported ONLY to quantify what the corrigendum changed. Not used for the
#     CCC validation itself.
def load_anderson2000():
    import importlib.util
    path = os.path.join(HERE, "anderson_benchmark_qc.py")
    if not os.path.exists(path):
        return None
    src = open(path).read()
    m = re.search(r"ANDERSON_TABLE2\s*=\s*\{(.*?)\n\}", src, re.S)
    if not m:
        return None
    ns = {}
    exec("ANDERSON_TABLE2 = {" + m.group(1) + "\n}", ns)
    return ns["ANDERSON_TABLE2"]


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print("CCC  vs  ANDERSON (2002 CORRIGENDUM) RMPS  --  effective collision strengths")
    print("=" * 78)
    print(f"  repo root : {ROOT}")
    print(f"  Upsilon   : {os.path.relpath(PDF_PATH, ROOT)}")
    print(f"  CCC sigma : {os.path.relpath(CCC_PATH, ROOT)}")
    print(f"  Eq.(3) prefactor 2*sqrt(pi)*alpha*c*a0^2 = {C_AND:.5e} cm^3/s "
          f"(paper quotes 2.1716e-8)")

    print("\n[1] Parsing Anderson 2002 table 1 from PDF")
    and02 = parse_anderson2002(PDF_PATH)
    print(f"  parsed {len(and02)} transitions x {len(TE_AND)} temperatures")
    out_ups = os.path.join(OUT_DIR, "anderson2002_upsilon_parsed.csv")
    and02.to_csv(out_ups, index=False)
    print(f"  wrote {os.path.relpath(out_ups, ROOT)}")

    # ── [2] What did the corrigendum change? ─────────────────────────────────
    print("\n[2] Corrigendum impact: Anderson 2002 vs Anderson 2000 (Upsilon ratio)")
    and00 = load_anderson2000()
    if and00 is None:
        print("  [SKIP] could not read the 2000 table from anderson_benchmark_qc.py")
    else:
        rel = []
        for _, r in and02.iterrows():
            key = (int(r.i_upper), int(r.j_lower))
            if key not in and00:
                continue
            for k, Te in enumerate(TE_AND):
                u00 = and00[key][k]
                u02 = r[f"ups_{Te:g}eV"]
                rel.append({"i": key[0], "j": key[1], "Te": Te,
                            "ups_2000": u00, "ups_2002": u02,
                            "pct": (u02 / u00 - 1.0) * 100.0})
        rel = pd.DataFrame(rel)
        print(f"  compared {rel.i.nunique() and len(rel)} points "
              f"({len(and00)} transitions matched)")
        print(f"    {'Te [eV]':>8}  {'mean|d|':>8}  {'max|d|':>8}  {'worst transition':>18}")
        for Te in TE_AND:
            s = rel[rel.Te == Te]
            w = s.loc[s.pct.abs().idxmax()]
            lab = (f"{nl_label(*IDX_TO_NL[int(w.j)])}->"
                   f"{nl_label(*IDX_TO_NL[int(w.i)])}")
            print(f"    {Te:8.1f}  {s.pct.abs().mean():7.2f}%  "
                  f"{s.pct.abs().max():7.2f}%  {lab:>12s} {w.pct:+7.1f}%")
        thesis = rel[rel.Te.isin(THESIS_TE)]
        print(f"\n  Over the thesis range (Te = 1,3,5,10 eV): mean |change| = "
              f"{thesis.pct.abs().mean():.2f}%, max = {thesis.pct.abs().max():.2f}%")
        print(f"  Anderson et al state the change is <~10% below 15 eV; here "
              f"{(thesis.pct.abs() < 10).mean()*100:.1f}% of points satisfy that.")
        rel.to_csv(os.path.join(OUT_DIR, "anderson_2000_vs_2002_upsilon.csv"), index=False)

    # ── [3] Anchor spot checks ───────────────────────────────────────────────
    print("\n[3] Anchor spot checks on Eq.(3) (independent of CCC)")
    for i_t, j_t, lab, Te, lo, hi in [(3, 1, "1s->2p", 1.0, 6e-13, 9e-13),
                                      (2, 1, "1s->2s", 1.0, 3e-13, 5e-13),
                                      (5, 2, "2s->3p", 1.0, 1e-8, 3e-8)]:
        n_up, _    = IDX_TO_NL[i_t]
        n_lo, l_lo = IDX_TO_NL[j_t]
        ups = and02.loc[(and02.i_upper == i_t) & (and02.j_lower == j_t),
                        f"ups_{Te:g}eV"].iloc[0]
        K = K_from_upsilon(ups, n_lo, l_lo, n_up, Te)
        print(f"    {lab:8s} Te={Te:.1f} eV  Ups={ups:.4f}  K={K:.4e} cm^3/s  "
              f"expect [{lo:.0e},{hi:.0e}]  {'OK' if lo <= K <= hi else 'OUT OF RANGE'}")

    # ── [4] Main comparison ──────────────────────────────────────────────────
    print("\n[4] CCC Maxwell-averaged K vs Anderson 2002 K")
    ccc = pd.read_csv(CCC_PATH)
    grp = ccc.groupby(["n_i", "l_i", "n_f", "l_f"])
    keys = set(grp.groups.keys())
    print(f"  CCC file holds {len(keys)} (n,l)->(n',l') blocks, {len(ccc)} sigma points")

    rows, missing = [], []
    for _, r in and02.iterrows():
        i_t, j_t = int(r.i_upper), int(r.j_lower)
        n_up, l_up = IDX_TO_NL[i_t]
        n_lo, l_lo = IDX_TO_NL[j_t]
        label = f"{nl_label(n_lo, l_lo)}->{nl_label(n_up, l_up)}"
        dE = threshold_eV(n_lo, n_up)

        key = (n_lo, l_lo, n_up, l_up)            # CCC stores excitation direction
        if key not in keys:
            missing.append(label)
            continue
        sub    = grp.get_group(key).sort_values("E_eV")
        E_ccc  = sub.E_eV.values
        s_ccc  = sub.sigma_a0sq.values
        E_grid = np.linspace(dE + 1e-4, E_ccc.max(), 5000)
        s_grid = np.interp(E_grid, E_ccc, s_ccc, left=0.0, right=0.0)

        for k in THESIS_TE_IDX:
            Te    = TE_AND[k]
            ups   = r[f"ups_{Te:g}eV"]
            K_and = K_from_upsilon(ups, n_lo, l_lo, n_up, Te)
            K_ccc = K_maxwell(s_grid, E_grid, Te)
            rows.append({
                "label": label, "i_upper": i_t, "j_lower": j_t,
                "n_lower": n_lo, "l_lower": l_lo, "n_upper": n_up, "l_upper": l_up,
                "Te_eV": Te, "dE_eV": dE, "Upsilon_2002": ups,
                "K_CCC": K_ccc, "K_Anderson2002": K_and,
                "ratio": K_ccc / K_and, "pct_err": (K_ccc / K_and - 1.0) * 100.0,
                "dn": n_up - n_lo, "dl": abs(l_up - l_lo),
                "dipole": abs(l_up - l_lo) == 1,
                "class": "ground_exc" if n_lo == 1 else "excited_exc",
                "E_max_CCC_eV": E_ccc.max(),
            })

    df = pd.DataFrame(rows)
    print(f"  matched {df.label.nunique()} / {len(and02)} Anderson transitions "
          f"({len(missing)} absent from CCC{': ' + ', '.join(missing) if missing else ''})")
    print(f"  {len(df)} (transition, Te) comparison points at Te = "
          f"{', '.join(f'{t:g}' for t in THESIS_TE)} eV")

    def block(sub, name):
        if len(sub) == 0:
            print(f"\n  {name}: no points")
            return None
        e = sub.pct_err.abs()
        w20, mean_e = (e < 20).mean() * 100, e.mean()
        verdict = ("PASS" if w20 > 85 and mean_e < 15 else
                   "PARTIAL" if w20 > 70 else "FAIL")
        worst = sub.loc[e.idxmax()]
        print(f"\n  {name}  (n={len(sub)})")
        print(f"    within 10% / 15% / 20% : {(e<10).mean()*100:5.1f}% / "
              f"{(e<15).mean()*100:5.1f}% / {w20:5.1f}%")
        print(f"    mean |err| {mean_e:6.2f}%   median |err| {e.median():6.2f}%   "
              f"median signed {sub.pct_err.median():+6.2f}%")
        print(f"    max |err|  {e.max():6.2f}%  ({worst.label} @ {worst.Te_eV:g} eV)")
        print(f"    VERDICT: {verdict}")
        return {"subset": name, "n": len(sub), "within10": (e < 10).mean() * 100,
                "within20": w20, "mean_abs_err": mean_e,
                "median_abs_err": e.median(), "max_abs_err": e.max(),
                "verdict": verdict}

    print("\n" + "=" * 78)
    print("SUMMARY  (CCC / Anderson-2002, Te = 1, 3, 5, 10 eV)")
    print("=" * 78)
    summ = [block(df, "ALL matched transitions"),
            block(df[df.n_upper <= 4], "n_upper <= 4  (CR-relevant core)"),
            block(df[(df.n_upper <= 4) & (df["class"] == "excited_exc")],
                  "n_upper <= 4, excited-state only"),
            block(df[df.n_upper == 5], "n_upper == 5  (diffuse-orbital set)")]
    summ = pd.DataFrame([s for s in summ if s])

    print("\n  Per-temperature (all transitions):")
    for Te in THESIS_TE:
        s = df[df.Te_eV == Te].pct_err.abs()
        print(f"    Te={Te:5.1f} eV  n={len(s):3d}  within20%={(s<20).mean()*100:5.1f}%  "
              f"mean|err|={s.mean():6.2f}%")

    print("\n  Dipole (|dl|=1) vs non-dipole -- the corrigendum touched dipoles most:")
    for flag, name in [(True, "dipole    "), (False, "non-dipole")]:
        s = df[df.dipole == flag].pct_err.abs()
        print(f"    {name}  n={len(s):3d}  within20%={(s<20).mean()*100:5.1f}%  "
              f"mean|err|={s.mean():6.2f}%")

    print("\n  Thesis anchor transitions:")
    for n_lo, l_lo, n_up, l_up, nm in [(1, 0, 2, 1, "1s->2p (Ly-alpha driver)"),
                                       (1, 0, 3, 1, "1s->3p"),
                                       (2, 1, 3, 2, "2p->3d (key stepwise)"),
                                       (2, 0, 3, 1, "2s->3p"),
                                       (3, 2, 4, 3, "3d->4f")]:
        s = df[(df.n_lower == n_lo) & (df.l_lower == l_lo) &
               (df.n_upper == n_up) & (df.l_upper == l_up)].sort_values("Te_eV")
        if len(s) == 0:
            print(f"    {nm:26s} -- not matched")
            continue
        errs = "  ".join(f"{t:g}eV:{e:+6.1f}%" for t, e in zip(s.Te_eV, s.pct_err))
        print(f"    {nm:26s} {errs}")

    out_csv = os.path.join(OUT_DIR, "ccc_vs_anderson2002_benchmark.csv")
    df.to_csv(out_csv, index=False)
    summ.to_csv(os.path.join(OUT_DIR, "ccc_vs_anderson2002_summary.csv"), index=False)
    print(f"\n  wrote {os.path.relpath(out_csv, ROOT)}  ({len(df)} rows)")

    # ── [5] Stored K table (user's parser + compute_K_CCC.py) ────────────────
    #     Two questions, kept separate on purpose:
    #       (a) does the STORED table reproduce a fresh re-average of the same
    #           cross sections?  -> tests the parser / Maxwell-averaging pipeline
    #       (b) does the STORED table agree with Anderson 2002?  -> the physics
    print("\n" + "=" * 78)
    print("[5] STORED K TABLE  (K_CCC_exc_table.npy from compute_K_CCC.py)")
    print("=" * 78)
    ccc_dir   = os.path.join(ROOT, "data", "processed", "collisions", "ccc")
    K_path    = os.path.join(ccc_dir, "K_CCC_exc_table.npy")
    meta_path = os.path.join(ccc_dir, "K_CCC_metadata.csv")
    Te_path   = os.path.join(ccc_dir, "Te_grid.npy")

    if not all(os.path.exists(x) for x in (K_path, meta_path, Te_path)):
        print("  [SKIP] stored K table / metadata / Te grid not found -- "
              "run compute_K_CCC.py first.")
        df_st = None
    else:
        K_st    = np.load(K_path)
        Te_st   = np.load(Te_path)
        meta    = pd.read_csv(meta_path)
        print(f"  K_CCC_exc_table {K_st.shape}, metadata {meta.shape}, "
              f"Te grid {Te_st.shape}: {Te_st.min():g}–{Te_st.max():g} eV")
        if K_st.shape != (len(meta), len(Te_st)):
            raise ValueError(f"Shape mismatch: K {K_st.shape} vs "
                             f"({len(meta)}, {len(Te_st)}) from metadata/Te grid.")
        if not np.isfinite(K_st).all():
            raise ValueError("Stored K table contains NaN/Inf.")

        # Which Anderson temperatures actually exist on the stored grid?
        exact = {Te: int(np.argmin(np.abs(Te_st - Te)))
                 for Te in THESIS_TE
                 if np.min(np.abs(Te_st - Te)) < 1e-9 * max(1.0, Te)}
        interp_only = [Te for Te in THESIS_TE if Te not in exact]
        print(f"  Anderson Te on the stored grid exactly : "
              f"{', '.join(f'{t:g} eV' for t in exact) or 'none'}")
        print(f"  Anderson Te requiring interpolation    : "
              f"{', '.join(f'{t:g} eV' for t in interp_only) or 'none'}")
        if interp_only:
            near = ", ".join(f"{t:g}->{Te_st[np.argmin(np.abs(Te_st - t))]:.3f}"
                             for t in interp_only)
            print(f"    (nearest stored grid points: {near} eV — nearest-neighbour")
            print(f"     matching would compare different temperatures; this script")
            print(f"     log-log interpolates in Te instead and flags those rows.)")

        idx_of = {(int(r.n_i), int(r.l_i), int(r.n_f), int(r.l_f)): int(r.idx)
                  for _, r in meta.iterrows()}

        st_rows, st_missing = [], []
        for _, r in and02.iterrows():
            i_t, j_t   = int(r.i_upper), int(r.j_lower)
            n_up, l_up = IDX_TO_NL[i_t]
            n_lo, l_lo = IDX_TO_NL[j_t]
            key = (n_lo, l_lo, n_up, l_up)
            label = f"{nl_label(n_lo, l_lo)}->{nl_label(n_up, l_up)}"
            if key not in idx_of:
                st_missing.append(label)
                continue
            Krow = K_st[idx_of[key], :]
            for k in THESIS_TE_IDX:
                Te  = TE_AND[k]
                ups = r[f"ups_{Te:g}eV"]
                if Te in exact:
                    K_stored, how = Krow[exact[Te]], "exact"
                else:
                    K_stored = float(np.exp(np.interp(np.log(Te), np.log(Te_st),
                                                      np.log(Krow))))
                    how = "interp"
                K_and = K_from_upsilon(ups, n_lo, l_lo, n_up, Te)
                st_rows.append({"label": label, "Te_eV": Te, "Te_match": how,
                                "n_lower": n_lo, "n_upper": n_up,
                                "K_stored": K_stored, "K_Anderson2002": K_and,
                                "pct_err": (K_stored / K_and - 1.0) * 100.0})
        df_st = pd.DataFrame(st_rows)
        print(f"  matched {df_st.label.nunique()} / {len(and02)} Anderson "
              f"transitions ({len(st_missing)} absent from the stored table)")

        # (a) stored vs freshly re-averaged, at exactly-matching Te only
        print("\n  (a) Pipeline self-consistency: stored K vs fresh re-average "
              "of the same sigma")
        merged = df_st[df_st.Te_match == "exact"].merge(
            df[["label", "Te_eV", "K_CCC"]], on=["label", "Te_eV"], how="inner")
        if len(merged) == 0:
            print("      [SKIP] no exactly-matching temperatures to compare on")
        else:
            d = (merged.K_stored / merged.K_CCC - 1.0) * 100.0
            print(f"      n={len(merged)} at Te = "
                  f"{', '.join(f'{t:g}' for t in sorted(merged.Te_eV.unique()))} eV")
            print(f"      max |diff| = {d.abs().max():.3f}%   "
                  f"median |diff| = {d.abs().median():.3f}%")
            w = merged.loc[d.abs().idxmax()]
            print(f"      worst: {w.label} @ {w.Te_eV:g} eV  "
                  f"stored={w.K_stored:.4e}  re-avg={w.K_CCC:.4e}  "
                  f"({d.abs().max():+.3f}%)")
            print("      VERDICT: " + ("CONSISTENT (<1%) — stored table reproduces "
                                       "a fresh average"
                                       if d.abs().max() < 1.0 else
                                       "DISCREPANT — stored table and fresh average "
                                       "disagree; investigate before use"))

        # (b) stored vs Anderson 2002
        print("\n  (b) Stored K vs Anderson 2002")
        for sub, name in [(df_st, "ALL matched"),
                          (df_st[df_st.n_upper <= 4], "n_upper <= 4"),
                          (df_st[df_st.n_upper == 5], "n_upper == 5")]:
            e = sub.pct_err.abs()
            print(f"      {name:14s} n={len(sub):3d}  within20%={(e<20).mean()*100:5.1f}%"
                  f"  mean|err|={e.mean():6.2f}%  median signed={sub.pct_err.median():+6.2f}%")
        df_st.to_csv(os.path.join(OUT_DIR, "storedK_vs_anderson2002.csv"), index=False)
        print(f"      wrote data/processed/collisions/storedK_vs_anderson2002.csv")

    # ── [6] Figure ───────────────────────────────────────────────────────────
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("CCC vs Anderson et al (2002 corrigendum) RMPS — Te = 1–10 eV",
                 fontsize=13, fontweight="bold")

    a = ax[0, 0]
    for Te, s in df.groupby("Te_eV"):
        a.scatter(s.K_Anderson2002, s.K_CCC, s=18, alpha=.75, label=f"Te={Te:g} eV")
    lim = [df[["K_Anderson2002", "K_CCC"]].values.min() * .5,
           df[["K_Anderson2002", "K_CCC"]].values.max() * 2]
    a.plot(lim, lim, "k-", lw=1.4, label="1:1")
    a.plot(lim, [x * 1.2 for x in lim], "k--", lw=.8, alpha=.6)
    a.plot(lim, [x * .8 for x in lim], "k--", lw=.8, alpha=.6, label="±20%")
    a.set(xscale="log", yscale="log", xlabel="K_Anderson2002 [cm³/s]",
          ylabel="K_CCC [cm³/s]", title="Rate coefficients, all matched transitions")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[0, 1]
    a.hist(df[df.n_upper <= 4].pct_err, bins=25, alpha=.65, color="steelblue",
           edgecolor="white", label="n_upper ≤ 4")
    a.hist(df[df.n_upper == 5].pct_err, bins=25, alpha=.65, color="darkorange",
           edgecolor="white", label="n_upper = 5")
    a.axvline(0, color="k", lw=1.4)
    for v in (-20, 20):
        a.axvline(v, color="r", ls="--", lw=1)
    a.set(xlabel="% error  (K_CCC/K_And − 1)×100", ylabel="count",
          title="Error distribution split by upper shell")
    a.legend(fontsize=9); a.grid(alpha=.3)

    a = ax[1, 0]
    for n_lo, l_lo, n_up, l_up, nm, c in [(1, 0, 2, 1, "1s→2p", "tab:blue"),
                                          (2, 1, 3, 2, "2p→3d", "tab:green")]:
        s = df[(df.n_lower == n_lo) & (df.l_lower == l_lo) &
               (df.n_upper == n_up) & (df.l_upper == l_up)].sort_values("Te_eV")
        if len(s):
            a.semilogy(s.Te_eV, s.K_CCC, "o-", color=c, ms=6, lw=2, label=f"{nm} CCC")
            a.semilogy(s.Te_eV, s.K_Anderson2002, "s--", color=c, ms=6, lw=2,
                       alpha=.6, label=f"{nm} Anderson02")
            a.fill_between(s.Te_eV, s.K_Anderson2002 * .8, s.K_Anderson2002 * 1.2,
                           color=c, alpha=.12)
    a.set(xlabel="Te [eV]", ylabel="K [cm³/s]",
          title="Anchor transitions (bands = Anderson ±20%)")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1, 1]
    e = np.sort(df.pct_err.abs().values)
    a.plot(e, np.arange(1, len(e) + 1) / len(e) * 100, "b-", lw=2, label="all")
    e4 = np.sort(df[df.n_upper <= 4].pct_err.abs().values)
    a.plot(e4, np.arange(1, len(e4) + 1) / len(e4) * 100, "g-", lw=2,
           label="n_upper ≤ 4")
    for t, c in [(10, "g"), (20, "r")]:
        a.axvline(t, color=c, ls="--", lw=1.2, alpha=.7)
    a.set(xlabel="|% error|", ylabel="cumulative % of comparisons", xlim=(0, 60),
          title="Cumulative error distribution")
    a.legend(fontsize=9); a.grid(alpha=.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ccc_vs_anderson2002_benchmark.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {os.path.relpath(fig_path, ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
