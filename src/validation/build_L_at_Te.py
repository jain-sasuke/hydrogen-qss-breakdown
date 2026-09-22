#!/usr/bin/env python
"""
build_L_at_Te.py
================
The CR operator L(Te, ne) and source S(Te, ne) at an ARBITRARY electron
temperature, built from the pipeline's own raw inputs and rate functions,
without editing any rate module.

WHY THIS EXISTS
---------------
Every rate table the pipeline stores is tabulated on the fixed 50-point Te grid
(1 to 10 eV, log-spaced), and assemble_cr_matrix.build_L takes a Te INDEX. The
thesis therefore has no operator between grid nodes: the ramp test of chapter 5
(sec 5.8.2) interpolates between two node operators, and
verify_physical_ramp_bound.py could only bracket the true interior operator
from a two-interval span (K17). Backlog K15 and K17 leave "L[Te(t)] at interior
temperatures" open pending a rate pipeline that accepts an arbitrary Te.

Every ingredient of L and S is either a closed-form function of Te that the
pipeline already exposes with a Te argument, or a Maxwell average of an
energy-resolved cross section that the pipeline stores in full. So the operator
at any Te can be built by calling those functions at that Te and re-doing the
Maxwell averages, and the build can be GATED by requiring it to reproduce the
stored tables and L_grid.npy / S_grid.npy at every one of the 50 x 8 nodes.

WHAT IS IMPORTED (read-only) AND WHAT IS RE-IMPLEMENTED
------------------------------------------------------
  compute_K_VS.py        K_exc_VS, K_deexc_DB, compute_f_pn_table  (import; Te argument)
  compute_K_TICS.py      maxwell_average_tics                        (import; scalar Te)
  ionization_rates.py    lotz_K_ion                                  (import; scalar Te)
  recombination_rates.py alpha_RR_nl, alpha_RR_shell, alpha_3BR_from_Kion (import; Te array)
  compute_lmix.py        compute_K_lmix(te_grid, ne_cm3)             (import; Te array)
  assemble_cr_matrix.py  build_L, build_source                       (import; fed one-Te slabs)
  radiative_rates        A_resolved, A_bund_res, A_bund_bund, gamma_* are Te-independent
                         and are LOADED from the stored .npy files, as build_L loads them.
  compute_K_CCC.py       CANNOT be imported: it reads the cross-section CSV, runs the
                         Maxwell average over all transitions and WRITES the tables at
                         module scope (lines 150-206, 364). Its maxwell_average,
                         threshold_eV, stat_weight and detailed_balance are re-implemented
                         here line for line, with its constants copied (ME, KB, A0_M, IH,
                         N_GRID), and the re-implementation is gated by bit-level
                         reproduction of K_CCC_exc_table.npy at all 50 nodes. The merge
                         rules of assemble_K_exc.py (CCC res-res for n_f <= 8; CCC summed
                         over l_f into the bundled n = 9 and n = 10 with detailed balance
                         on the shell weight 2n^2; Vriens-Smeets for res -> n = 11..15 and
                         bundled <-> bundled) are re-implemented likewise and gated
                         against K_exc_full.npy / K_deexc_full.npy.

The l-mixing block is built with the Debye cutoff FROZEN at compute_lmix.NE_DEFAULT
by default, because that is how L_grid.npy was built (K_lmix.npy has no density
axis; see verify_bundling_total_loss.py). Pass lmix_ne="local" to evaluate the
cutoff at the local density instead; that operator is NOT the thesis operator and
the gate does not apply to it.

GATE (run with --gate, and run automatically by any script that imports this)
-------------------------------------------------------------------------------
At every Te node t = 0..49: the rebuilt K_exc_full[:, :, t], K_deexc_full[:, :, t],
K_ion_final[:, t], alpha_RR_resolved[:, t], alpha_RR_bundled[:, t],
alpha_3BR_resolved[:, t], alpha_3BR_bundled[:, t], K_lmix[:, :, t] equal the stored
arrays to 1e-12 (relative to each array's largest entry), and at every (t, j)
L and S equal L_grid[t, j] and S_grid[t, j] to 1e-12 relative to the largest
entry of that operator. The CCC transition list rebuilt from the cross-section
CSV must equal K_CCC_metadata.csv row for row.

USAGE
-----
  python build_L_at_Te.py --gate
  python build_L_at_Te.py --te 2.5 --ne 1.389e14 --write [--out DIR] [--lmix frozen|local]

  from build_L_at_Te import OperatorBuilder
  ob = OperatorBuilder.load(); ob.gate()          # raises if the reproduction fails
  L, S = ob.operator(2.5, 1.389e14)               # (43, 43) s^-1, (43,) s^-1 per unit n_ion

OUTPUT (with --write): validation/L_at_Te/Te<value>_ne<value>/  L.npy, S.npy, build.txt
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, sys, contextlib
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE)
if str(ROOT / "src") not in sys.path: sys.path.insert(0, str(ROOT / "src"))   # assemble_cr_matrix does `import rates`

# ---- constants copied from compute_K_CCC.py (lines 77-81, 85); the gate below is what makes copying them safe
CCC_ME, CCC_KB, CCC_A0_M, CCC_IH, CCC_N_GRID = 9.10938e-31, 1.60218e-19, 5.29177e-11, 13.6058, 5000
ASSEMBLE_IH = 13.6058      # assemble_K_exc.py line 71, used in the bundled-collapse detailed balance


def _load(name: str, path: Path):
    """Import a pipeline module by file path, read-only. Nothing in it is edited or re-run."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(mod)
    return mod


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def ccc_maxwell_average(E_grid: np.ndarray, sig_grid: np.ndarray, Te: float) -> np.ndarray:
    """compute_K_CCC.maxwell_average, lines 101-127, on pre-interpolated grids (rows = transitions), one scalar Te.
    K = sqrt(8/pi/me) (kTe)^-3/2 * int sigma(E) E exp(-E/kTe) dE, trapezoid on the 5000-point grid, a0^2 eV^2 -> m^2 J^2 -> cm^3/s."""
    integrand = sig_grid * E_grid * np.exp(-E_grid / Te)
    integral_SI = np.trapezoid(integrand, E_grid, axis=-1) * CCC_A0_M**2 * CCC_KB**2
    return np.sqrt(8.0 / np.pi / CCC_ME) * (Te * CCC_KB) ** (-1.5) * integral_SI * 1e6


class OperatorBuilder:
    def __init__(self, root: Path):
        self.root = root; P = lambda *p: root.joinpath(*p)
        self.paths = dict(
            ccc_xs=P("data/processed/collisions/ccc/ccc_crosssections.csv"), ccc_meta=P("data/processed/collisions/ccc/K_CCC_metadata.csv"),
            ccc_exc=P("data/processed/collisions/ccc/K_CCC_exc_table.npy"), ccc_deexc=P("data/processed/collisions/ccc/K_CCC_deexc_table.npy"),
            tics_xs=P("data/processed/collisions/tics/tics_crosssections.csv"), vs_meta=P("data/processed/collisions/vs/K_VS_metadata.csv"),
            hb=P("data/processed/Radiative/H_A_E1_LS_n1_15_physical.csv"), state_index=P("data/processed/collisions/K_exc_full/state_index.csv"),
            K_exc_full=P("data/processed/collisions/K_exc_full/K_exc_full.npy"), K_deexc_full=P("data/processed/collisions/K_exc_full/K_deexc_full.npy"),
            K_exc_meta=P("data/processed/collisions/K_exc_full/K_exc_meta.csv"),
            K_ion_final=P("data/processed/collisions/tics/K_ion_final.npy"),
            alpha_RR_res=P("data/processed/recombination/alpha_RR_resolved.npy"), alpha_RR_bund=P("data/processed/recombination/alpha_RR_bundled.npy"),
            alpha_3BR_res=P("data/processed/recombination/alpha_3BR_resolved.npy"), alpha_3BR_bund=P("data/processed/recombination/alpha_3BR_bundled.npy"),
            A_resolved=P("data/processed/Radiative/A_resolved.npy"), A_bund_res=P("data/processed/Radiative/A_bund_res.npy"),
            A_bund_bund=P("data/processed/Radiative/A_bund_bund.npy"), gamma_resolved=P("data/processed/Radiative/gamma_resolved.npy"),
            gamma_bundled=P("data/processed/Radiative/gamma_bundled.npy"), K_lmix=P("data/processed/lmix/K_lmix.npy"),
            L_grid=P("data/processed/cr_matrix/L_grid.npy"), S_grid=P("data/processed/cr_matrix/S_grid.npy"),
            Te_grid_L=P("data/processed/cr_matrix/Te_grid_L.npy"), ne_grid_L=P("data/processed/cr_matrix/ne_grid_L.npy"),
            m_vs=P("src/rates/compute_K_VS.py"), m_tics=P("src/rates/compute_K_TICS.py"), m_ion=P("src/rates/ionization_rates.py"),
            m_rec=P("src/rates/recombination_rates.py"), m_lmix=P("src/rates/compute_lmix.py"), m_asm=P("src/rates/assemble_cr_matrix.py"),
            m_ccc=P("src/rates/compute_K_CCC.py"), m_kexc=P("src/rates/assemble_K_exc.py"))
        for q in self.paths.values():
            if not q.is_file(): raise FileNotFoundError(f"required input missing: {q}")
        self.vs = _load("compute_K_VS", self.paths["m_vs"]); self.tics = _load("compute_K_TICS", self.paths["m_tics"])
        self.ion = _load("ionization_rates", self.paths["m_ion"]); self.rec = _load("recombination_rates", self.paths["m_rec"])
        self.lm = _load("compute_lmix", self.paths["m_lmix"]); self.asm = _load("assemble_cr_matrix", self.paths["m_asm"])
        for mod, val in ((self.vs, self.vs.IH_eV), (self.tics, self.tics.IH_eV), (self.ion, self.ion.IH_EV), (self.rec, self.rec.IH_eV)):
            if val != CCC_IH: raise RuntimeError(f"{mod.__name__} uses IH = {val}, compute_K_CCC uses {CCC_IH}; the threshold conventions have diverged")

        # ---- state ordering, from the pipeline's file
        self.si = pd.read_csv(self.paths["state_index"]); nv = self.si.n.values.astype(int); lv = self.si.l.values.astype(int)
        self.nS = len(self.si); self.byn = {(int(n), int(l)): k for k, (n, l) in enumerate(zip(nv, lv))}
        self.n_res = int((lv >= 0).sum()); self.g = self.si.g.values.astype(float)
        if self.n_res != 36 or self.nS != 43: raise RuntimeError(f"state_index.csv: {self.n_res} resolved of {self.nS} states; assemble_cr_matrix.build_L hardcodes 36 and 43")

        # ---- CCC: transition list in compute_K_CCC's order (sorted groupby over excitation rows), pre-interpolated grids
        xs = pd.read_csv(self.paths["ccc_xs"]); exc = xs[xs.n_f > xs.n_i]
        grp = exc.groupby(["n_i", "l_i", "n_f", "l_f"], sort=True); keys = list(grp.groups.keys())
        meta = pd.read_csv(self.paths["ccc_meta"])
        if len(keys) != len(meta) or any((int(a), int(b), int(c), int(d)) != (int(r.n_i), int(r.l_i), int(r.n_f), int(r.l_f))
                                        for (a, b, c, d), r in zip(keys, meta.itertuples(index=False))):
            raise RuntimeError("CCC transition list rebuilt from ccc_crosssections.csv differs from K_CCC_metadata.csv")
        self.ccc_keys = [(int(a), int(b), int(c), int(d)) for (a, b, c, d) in keys]
        Eg = np.empty((len(keys), CCC_N_GRID)); Sg = np.empty_like(Eg); dEs = np.empty(len(keys)); om_i = np.empty(len(keys)); om_f = np.empty(len(keys))
        for r, key in enumerate(keys):
            gg = grp.get_group(key).sort_values("E_eV"); E_raw = gg.E_eV.values; sig_raw = gg.sigma_a0sq.values
            n_i, l_i, n_f, l_f = self.ccc_keys[r]; dE = CCC_IH * (1.0 / n_i**2 - 1.0 / n_f**2)      # compute_K_CCC.threshold_eV
            Eg[r] = np.linspace(dE + 1e-4, E_raw.max(), CCC_N_GRID); Sg[r] = np.interp(Eg[r], E_raw, sig_raw, left=0.0, right=0.0)
            dEs[r] = dE; om_i[r] = 2 * (2 * l_i + 1); om_f[r] = 2 * (2 * l_f + 1)                    # compute_K_CCC.stat_weight
        self.ccc_E, self.ccc_S, self.ccc_dE, self.ccc_om_i, self.ccc_om_f = Eg, Sg, dEs, om_i, om_f
        if np.abs(meta.dE_eV.values - dEs).max() > 1e-5: raise RuntimeError("CCC thresholds differ from K_CCC_metadata.csv by more than its printed precision")

        # ---- TICS: resolved (n, l) and the bundled n = 9 row, raw arrays sorted by energy (compute_K_TICS lines 132-172)
        tx = pd.read_csv(self.paths["tics_xs"]); res = tx[tx.type == "resolved"]; bund = tx[tx.type == "bundled"]
        self.tics_res = {}
        for (n, l), k in self.byn.items():
            if l < 0: continue
            sub = res[(res.n == n) & (res.l == l)].sort_values("E_eV")
            if len(sub) == 0: raise RuntimeError(f"no TICS data for n={n}, l={l}")
            self.tics_res[k] = (sub.E_eV.values, sub.sigma_a0sq.values, CCC_IH / n**2)
        sub9 = bund[bund.n == 9].sort_values("E_eV")
        if len(sub9) == 0: raise RuntimeError("TICS.9 bundled row absent; ionization_rates.py used it directly")
        self.tics_n9 = (sub9.E_eV.values, sub9.sigma_a0sq.values, CCC_IH / 81.0)

        # ---- Vriens-Smeets: the pipeline's transition list and its oscillator-strength table (compute_K_VS lines 236-246)
        self.vs_meta = pd.read_csv(self.paths["vs_meta"]); self.f_table = self.vs.compute_f_pn_table(str(self.paths["hb"]))
        self.vs_rows = []
        for r in self.vs_meta.itertuples(index=False):
            p, lp, n, ln = int(r.p), int(r.l_p), int(r.n), int(r.l_n)
            if n == 9 and lp >= 0: continue                                                          # assemble_K_exc Block 4: res -> n9 comes from CCC
            f_pn = self.f_table.get((p, n), None); src = "HoangBinh"
            if f_pn is None or f_pn <= 0:
                f_pn = max((32 / (3 * np.sqrt(3) * np.pi)) * (p**5 * n) / (n**2 - p**2) ** 3, 1e-10); src = "Kramers_fallback"
            if src != r.f_source: raise RuntimeError(f"VS f_pn source for ({p},{n}) rebuilt as {src}, metadata says {r.f_source}")
            self.vs_rows.append((self.byn[(p, lp)], self.byn[(n, ln)], p, n, float(f_pn), float(r.g_p), float(r.g_n), int(r.idx)))
        if len(self.vs_rows) != 201: raise RuntimeError(f"{len(self.vs_rows)} VS pairs after excluding res->n9; assemble_K_exc records 201")

        # ---- Te-independent arrays, loaded as build_L loads them
        self.rad = {k: np.load(self.paths[k]) for k in ("A_resolved", "A_bund_res", "A_bund_bund", "gamma_resolved", "gamma_bundled")}
        self._cache = {}

    @classmethod
    def load(cls, root: str | Path | None = None) -> "OperatorBuilder":
        return cls(Path(root).resolve() if root else ROOT)

    # ------------------------------------------------------------------ the Te-dependent tables at one scalar Te
    def tables(self, Te: float, lmix_ne: float | None = None) -> dict:
        """All rate arrays assemble_cr_matrix.build_L needs, as one-Te slabs (last axis length 1)."""
        Te = float(Te); key = (Te, lmix_ne)
        if key in self._cache: return self._cache[key]
        nS, nr = self.nS, self.n_res; TeA = np.array([Te])
        # CCC excitation and de-excitation at this Te (compute_K_CCC lines 101-149)
        Kx = ccc_maxwell_average(self.ccc_E, self.ccc_S, Te); Kd = Kx * (self.ccc_om_i / self.ccc_om_f) * np.exp(self.ccc_dE / Te)
        Ke = np.zeros((nS, nS)); Kde = np.zeros((nS, nS))
        b9, b10 = self.byn[(9, -1)], self.byn[(10, -1)]; sum9 = np.zeros(nr); sum10 = np.zeros(nr)
        for r, (n_i, l_i, n_f, l_f) in enumerate(self.ccc_keys):
            if n_f <= 8: si_, sf_ = self.byn[(n_i, l_i)], self.byn[(n_f, l_f)]; Ke[si_, sf_] = Kx[r]; Kde[sf_, si_] = Kd[r]      # Block 1
            elif n_f == 9: sum9[self.byn[(n_i, l_i)]] += Kx[r]                                                                  # Block 2, summed over l_f
            elif n_f == 10 and n_i <= 8: sum10[self.byn[(n_i, l_i)]] += Kx[r]                                                    # Block 3 (K_exc_to_n10_bundled)
        for s_ in range(nr):
            n_i = int(self.si.n[s_]); g_i = self.g[s_]
            Ke[s_, b9] = sum9[s_]; Kde[b9, s_] = sum9[s_] * (g_i / self.g[b9]) * np.exp(ASSEMBLE_IH * (1 / n_i**2 - 1 / 81) / Te)
            Ke[s_, b10] = sum10[s_]; Kde[b10, s_] = sum10[s_] * (g_i / self.g[b10]) * np.exp(ASSEMBLE_IH * (1 / n_i**2 - 1 / 100) / Te)
        # Vriens-Smeets (compute_K_VS.K_exc_VS / K_deexc_DB at this Te), assemble_K_exc Block 4
        for (si_, sf_, p, n, f_pn, g_p, g_n, _) in self.vs_rows:
            ke = self.vs.K_exc_VS(p, n, f_pn, TeA, g_p=g_p, g_n=g_n); E_pn = self.vs.IH_eV * (1.0 / p**2 - 1.0 / n**2)
            Ke[si_, sf_] = ke[0]; Kde[sf_, si_] = self.vs.K_deexc_DB(ke, g_p, g_n, E_pn, TeA)[0]
        # ionisation (compute_K_TICS.maxwell_average_tics; ionization_rates Blocks 1-3)
        Kion = np.zeros(nS)
        for k, (E, sg, I_n) in self.tics_res.items(): Kion[k] = self.tics.maxwell_average_tics(E, sg, I_n, Te)
        Kion[b9] = self.tics.maxwell_average_tics(*self.tics_n9, Te)
        for n in range(10, 16): Kion[self.byn[(n, -1)]] = self.ion.lotz_K_ion(n, Te)
        # recombination (recombination_rates.compute_recombination_rates, resolved / n = 9 / n = 10..15 blocks)
        aRR_r = np.zeros(nr); a3_r = np.zeros(nr); aRR_b = np.zeros(nS - nr); a3_b = np.zeros(nS - nr)
        for k in range(nr):
            n, l = int(self.si.n[k]), int(self.si.l[k])
            aRR_r[k] = self.rec.alpha_RR_nl(n, l, TeA)[0]; a3_r[k] = self.rec.alpha_3BR_from_Kion(Kion[k:k + 1], 2 * (2 * l + 1), self.rec.IH_eV / n**2, TeA)[0]
        for n in range(9, 16):
            k = self.byn[(n, -1)]; b = k - nr
            aRR_b[b] = self.rec.alpha_RR_shell(n, TeA)[0]; a3_b[b] = self.rec.alpha_3BR_from_Kion(Kion[k:k + 1], 2 * n**2, self.rec.IH_eV / n**2, TeA)[0]
        # proton l-mixing (compute_lmix.compute_K_lmix), frozen cutoff unless a local density is given
        with contextlib.redirect_stdout(io.StringIO()):
            Kl = self.lm.compute_K_lmix(te_grid=TeA, ne_cm3=self.lm.NE_DEFAULT if lmix_ne is None else float(lmix_ne))
        out = dict(K_exc_full=Ke[:, :, None], K_deexc_full=Kde[:, :, None], K_ion_final=Kion[:, None],
                   alpha_RR_res=aRR_r[:, None], alpha_RR_bund=aRR_b[:, None], alpha_3BR_res=a3_r[:, None], alpha_3BR_bund=a3_b[:, None],
                   K_lmix=Kl, **self.rad)
        self._cache[key] = out; return out

    def operator(self, Te: float, ne: float, lmix_ne: float | None = None):
        """L (43, 43) [s^-1] and S (43,) [s^-1 per unit n_ion] at (Te, ne), through assemble_cr_matrix.build_L / build_source."""
        t = self.tables(Te, lmix_ne)
        return self.asm.build_L(0, float(ne), t), self.asm.build_source(0, float(ne), t, n_ion=1.0)

    # ------------------------------------------------------------------ the gate
    def gate(self, tol: float = 1e-12, verbose: bool = True) -> dict:
        te = np.load(self.paths["Te_grid_L"]); ne = np.load(self.paths["ne_grid_L"]); L = np.load(self.paths["L_grid"]); S = np.load(self.paths["S_grid"])
        stored = dict(K_exc_full=np.load(self.paths["K_exc_full"]), K_deexc_full=np.load(self.paths["K_deexc_full"]), K_ion_final=np.load(self.paths["K_ion_final"]),
                      alpha_RR_res=np.load(self.paths["alpha_RR_res"]), alpha_RR_bund=np.load(self.paths["alpha_RR_bund"]),
                      alpha_3BR_res=np.load(self.paths["alpha_3BR_res"]), alpha_3BR_bund=np.load(self.paths["alpha_3BR_bund"]), K_lmix=np.load(self.paths["K_lmix"]))
        ccc_exc = np.load(self.paths["ccc_exc"]); ccc_deexc = np.load(self.paths["ccc_deexc"])
        worst = {k: 0.0 for k in list(stored) + ["K_CCC_exc", "K_CCC_deexc", "L", "S"]}
        for t, Te in enumerate(te):
            Kx = ccc_maxwell_average(self.ccc_E, self.ccc_S, float(Te)); Kd = Kx * (self.ccc_om_i / self.ccc_om_f) * np.exp(self.ccc_dE / Te)
            worst["K_CCC_exc"] = max(worst["K_CCC_exc"], np.abs(Kx - ccc_exc[:, t]).max() / np.abs(ccc_exc[:, t]).max())
            worst["K_CCC_deexc"] = max(worst["K_CCC_deexc"], np.abs(Kd - ccc_deexc[:, t]).max() / np.abs(ccc_deexc[:, t]).max())
            tb = self.tables(float(Te))
            for k, ref in stored.items():
                mine = tb[k][..., 0]; ref_t = ref[..., t]
                worst[k] = max(worst[k], np.abs(mine - ref_t).max() / np.abs(ref_t).max())
            for j, nej in enumerate(ne):
                Lm, Sm = self.operator(float(Te), float(nej))
                worst["L"] = max(worst["L"], np.abs(Lm - L[t, j]).max() / np.abs(L[t, j]).max())
                worst["S"] = max(worst["S"], np.abs(Sm - S[t, j]).max() / np.abs(S[t, j]).max())
        self._cache.clear()
        if verbose:
            print(f"  GATE: rebuilt tables vs stored, worst relative difference over {len(te)} Te nodes" + (f" x {len(ne)} densities" if True else ""))
            for k, v in worst.items(): print(f"    {k:14s} {v:.2e}")
        bad = {k: v for k, v in worst.items() if not (v <= tol)}
        if bad: raise RuntimeError(f"build_L_at_Te gate FAILED (tolerance {tol:.0e}): " + ", ".join(f"{k} {v:.2e}" for k, v in bad.items()))
        if verbose: print(f"  GATE PASSED at {tol:.0e}: the arbitrary-Te build reproduces every stored table, L_grid and S_grid at all nodes.")
        return worst

    def provenance(self) -> list[str]:
        return [f"# {q.relative_to(self.root)} sha256 {sha(q)}" for q in self.paths.values()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--gate", action="store_true", help="reproduce every stored table and L_grid/S_grid at all 50 x 8 nodes")
    ap.add_argument("--te", type=float, default=None); ap.add_argument("--ne", type=float, default=None)
    ap.add_argument("--lmix", choices=("frozen", "local"), default="frozen")
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ob = OperatorBuilder.load()
    print("=" * 90); print("build_L_at_Te: the CR operator at an arbitrary electron temperature"); print(f"generated {datetime.now():%Y-%m-%d %H:%M}  interpreter {sys.executable}"); print("=" * 90)
    if a.gate or a.te is None:
        ob.gate()
    if a.te is not None:
        if a.ne is None: raise SystemExit("--ne is required with --te")
        ctx = CRContext.load(root=ROOT)
        if not (ctx.te_grid.min() <= a.te <= ctx.te_grid.max()): raise SystemExit(f"Te = {a.te} eV lies outside the grid {ctx.te_grid.min()}..{ctx.te_grid.max()} eV; the cross sections are not validated there")
        lm_ne = None if a.lmix == "frozen" else a.ne
        L, S = ob.operator(a.te, a.ne, lm_ne)
        ev = np.linalg.eigvals(L); lam0 = ev[np.argmin(np.abs(ev))]; u = (-np.linalg.solve(L, S))[ctx.ground_index]
        print(f"\n  L({a.te} eV, {a.ne:.4g} cm^-3), l-mixing cutoff {a.lmix}: tau_slow = {1/abs(lam0.real):.6e} s, u_CRE = {u:.6e}, column-sum residual vs -K_ion ne: "
              f"{np.abs(L.sum(axis=0) + ob.tables(a.te, lm_ne)['K_ion_final'][:, 0] * a.ne).max() / np.abs(L).max():.1e}")
        if a.write:
            out = Path(a.out) if a.out else ROOT / "validation" / "L_at_Te" / f"Te{a.te:g}_ne{a.ne:g}" ; out.mkdir(parents=True, exist_ok=True)
            np.save(out / "L.npy", L); np.save(out / "S.npy", S)
            hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}  numpy {np.__version__}",
                   f"# Te = {a.te!r} eV, ne = {a.ne!r} cm^-3, l-mixing Debye cutoff: {a.lmix}" + ("" if a.lmix == "local" else f" (compute_lmix.NE_DEFAULT = {ob.lm.NE_DEFAULT:g})"),
                   "# gate: rebuilt tables reproduce the stored ones at all 50 x 8 nodes (run --gate to print the worst differences)"] + ob.provenance()
            (out / "build.txt").write_text("\n".join(hdr) + f"\n# tau_slow {1/abs(lam0.real):.10e} s\n# u_CRE {u:.10e}\n")
            print(f"  wrote {out.relative_to(ROOT)}/L.npy, S.npy, build.txt")
    return 0


if __name__ == "__main__": sys.exit(main())
