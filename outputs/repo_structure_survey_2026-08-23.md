> ## ⚠️ STALE SNAPSHOT — READ BEFORE USING
>
> **This document surveys the repository as it stood on 2026-08-23, not as it
> stands now.** It was produced on 2026-09-10 by agents whose filesystem view was
> frozen ~18 days behind the working tree. The staleness was detected after the
> document was written and is recorded here rather than silently corrected.
>
> **What is still true:** every mtime quoted here was re-checked against the live
> filesystem and matches (`L_grid.npy` 2026-07-21 20:44:53;
> `validation/{M_grid,tau_QSS_grid,tau_relax_grid}.npy` and `gate_{B,C,D}.csv`
> 2026-08-23 11:17:01). The relative orderings, the write collisions, the quoted
> code and the error-metric inventory are a correct account of the 2026-08-23 tree.
>
> **What is wrong:**
> - The inventory covers **66** `.py` files under `src/`. There are now **80**.
>   Fourteen are missing — listed below.
> - Anywhere this document says a file was regenerated "today", it means
>   **2026-08-23**, eighteen days ago. It does not mean the current session.
> - Items 2, 7, 9 and 10 draw conclusions from a script census that is now
>   incomplete, so their negative findings ("no reader found", "no producer
>   found", "59 of 66 do not import cr_context") are **not safe to rely on**.
>
> **Scripts absent from this survey** (all postdate the snapshot):
>
> | path | mtime |
> |---|---|
> | `src/validation/verify_fujimoto_table41.py` | 2026-08-23 15:38:47 |
> | `src/validation/verify_ch3_claims.py` | 2026-08-24 00:50:35 |
> | `src/validation/verifych3_gb.py` | 2026-08-24 01:51:25 |
> | `src/analysis/make_ch3_figures.py` | 2026-08-24 02:10:19 |
> | `src/validation/verify_eps_gridmap.py` | 2026-08-24 02:17:27 |
> | `src/validation/grid.py` | 2026-08-24 02:55:23 |
> | `src/validation/verify_ridge_mechanism.py` | 2026-08-24 03:20:58 |
> | `src/validation/preflight.py` | 2026-08-24 03:40:09 |
> | `src/validation/audit_writers.py` | 2026-08-24 03:50:02 |
> | `src/validation/verify_plateau_bridge.py` | 2026-08-24 04:52:34 |
> | `src/validation/verify_lyman_trapping.py` | 2026-09-10 00:05:19 |
> | `src/analysis/make_ch5_figures.py` | 2026-09-10 01:00:16 |
> | `src/validation/verify_reservoir_gain.py` | 2026-09-10 10:13:28 |
> | `src/analysis/make_story_figures.py` | 2026-09-10 18:16:57 |
>
> Two of these bear directly on this document's own findings:
> `audit_writers.py` exists to check the one-writer-per-output-path rule that
> Item 2 reports as unaddressed, and `grid.py` is a second grid-provenance module
> alongside `cr_context.py`, which Item 7 treats as the sole canonical source.
>
> — appended 2026-09-10 18:17

# Repository structure survey — 2026-08-23

**Scope:** static survey of what this repo computes and what depends on what.
**Method:** four read-only verifier agents. Nothing was executed, no project script
was run or imported, no file in the repo was modified. All inspection via
`find`, `grep`, `sed -n`, `stat -f`, `cat`, `wc`, `od`, `strings`, `diff`, `comm`,
`git log`/`git show`. macOS/BSD userland.

**Protocol note:** one agent wrote three temporary `.txt` files to the session
scratchpad while building item 7's file list, then deleted them (`rm -f`) and
switched to pure pipes. Nothing was written inside the repo. Recorded here because
the instruction was to write nothing anywhere.

**This document is a structural survey. It contains no physics interpretation.**

---

## Contents

1. [Script inventory](#item-1--script-inventory)
2. [Write collisions](#item-2--write-collisions)
3. [Dependency edges](#item-3--dependency-edges)
4. [What depends on L_grid.npy](#item-4--what-depends-on-l_gridnpy)
5. [The Balmer observable](#item-5--the-balmer-observable)
6. [Error metric inventory](#item-6--error-metric-inventory)
7. [Hardcoded grid assumptions](#item-7--hardcoded-grid-assumptions)
8. [ADAS / Gate D](#item-8--adas--gate-d)
9. [Figure provenance](#item-9--figure-provenance)
10. [Orphans](#item-10--orphans)
11. [Document-vs-disk conflicts](#document-vs-disk-conflicts)

---

## ITEM 1 — SCRIPT INVENTORY

Ordering command:

```
find src -name '*.py' | while read f; do stat -f '%Sm %N' -t '%Y-%m-%d %H:%M' "$f"; done | sort -r
```

66 files (`find src -name '*.py' | wc -l` = 66). Per-file extraction:

```
grep -nE "np\.load\(|np\.loadtxt\(|pd\.read_csv\(|open\(|json\.load\(|h5py\.|loadmat\(|np\.save\(|np\.savez\(|to_csv\(|savefig\(|json\.dump\(|np\.savetxt\(|cr_context" <file>
```

plus a `grep -n` for the defining assignment (`_REPO =`, `PATHS = {`, `OUT_DIR =`)
whenever the load/write call used a variable rather than a literal.

| # | Path | Docstring line 1 | Loads | Writes | mtime |
|---|---|---|---|---|---|
|1|`src/validation/verify_divertor_map.py`|"verify_divertor_map.py"|cr_context -> `cr_matrix/{L_grid,Te_grid_L,ne_grid_L}.npy` + state_index.csv; `cr_matrix/S_grid.npy`|`validation/divertor_map/divertor_map.{txt,csv}`|08-23 11:18|
|2|`src/validation/verify_plateau_gridmap.py`|"verify_plateau_gridmap.py"|same + `S_grid.npy`|`validation/plateau_gridmap/plateau_gridmap.{txt,csv}`|08-23 00:22|
|3|`src/validation/verify_plateau_slowmode.py`|"verify_plateau_slowmode.py"|same + `S_grid.npy`|`validation/plateau_slowmode/plateau_slowmode.{txt,csv}`|08-23 00:15|
|4|`src/validation/validate_gates.py`|"validate_gates.py"|`PATHS`: L_grid, S_grid, Te_grid_L, ne_grid_L, `collisions/tics/K_ion_final.npy`, `K_exc_full/{K_exc_full,K_deexc_full}.npy`, `K_exc_meta.csv`, `adas/{SCD96,ACD96}_interpolated.csv`|`validation/gate_{B,C,D}.csv`, `gate_summary.txt`, **+ `M_grid.npy`, `tau_QSS_grid.npy`, `tau_relax_grid.npy`** (if `'M_grid' in r`, Gate E)|08-22 14:28|
|5|`src/validation/qss_analysis.py`|"qss_analysis.py"|`PATHS`: L_grid, S_grid, Te_grid, ne_grid|`validation/` -> `M_grid.npy`, `tau_QSS_grid.npy`, `tau_relax_grid.npy`, `epsilon_traces.npz`, `breakdown_map.csv`, `qss_analysis_summary.txt`|08-22 14:28|
|6|`src/validation/verify_timescales.py`|"verify_timescales.py — Independent verification of the two-timescale structure…"|cr_context|`validation/{spectrum_testpoint,timescale_verification}.csv`|08-22 11:33|
|7|`src/validation/verify_ramp_vs_step.py`|"verify_ramp_vs_step.py — Does the step picture apply at divertor timescales?"|cr_context; `S_grid.npy`|`validation/ramp_vs_step.csv`|08-22 11:33|
|8|`src/validation/verify_boundary_descent.py`|"verify_boundary_descent.py — Does the relaxation eigenmode descend in n with density?"|cr_context only|`validation/boundary_descent.csv`, `{outstem}.{png,pdf}`|08-22 11:33|
|9|`src/validation/cr_context.py`|"cr_context.py — Load the CR model's grids and state ordering from the codebase"|`Te_grid_L.npy`, `ne_grid_L.npy`, `L_grid.npy`; `K_exc_full/state_index.csv` (fallback `Radiative/state_index.csv`)|— (library)|08-22 11:33|
|10|`src/rates/compute_lmix.py`|"compute_lmix.py"|none — analytic PSM20 Debye; `TE_GRID` local `np.logspace`|`data/processed/lmix/K_lmix.npy`|07-21 20:44|
|11|`src/parsers/qc_ccc.py`|"qc_ccc.py"|`sys.argv[1]` csv|`{argv[2]}/ccc_qc_report.png`|06-14 14:54|
|12|`src/rates/S_criterion_3P.py`|"S_criterion_3P.py"|`cr_matrix/{L_grid,S_grid}.npy`|`figures/S_3P_{map,collapse}.*`; `sensitivity/{S_3P_grid,a_3P_grid}.npy`, `eps_3P_results.npz`|05-20 23:28|
|13|`src/validation/verify_bundling_psm20.py`|"verify_bundling_psm20.py"|`Te_grid_L.npy`, `ne_grid_L.npy`, `lmix/K_lmix.npy` (+3 fallback paths)|`outputs/bundling/{K_lmix_per_shell.npy,bundling_ratio.npy,bundling_report.txt}`|05-20 22:56|
|14|`src/rates/Balmer_timescale_audit.py`|"Balmer_timescale_audit.py"|`sensitivity/regime/Balmer_transient_regime_heatmap_DTe_p0p60.csv`|`sensitivity/timescale_audit/*.csv`; `figures/timescale_audit/` 4 figs|05-10 05:36|
|15|`src/rates/plot_regime_persistence_fixed.py`|"plot_regime_persistence_fixed.py"|same regime CSV (default)|`out_csv` (derived at call site); `figures/regime/{stem}.*`|05-10 05:29|
|16|`src/rates/Balmer_transient_regime_heatmap.py`|"Balmer_transient_regime_heatmap.py"|re-reads its own `out_csv` for incremental append|`sensitivity/regime/Balmer_transient_regime_heatmap_DTe_{slug}.csv`; `figures/regime/`|05-10 05:19|
|17|`src/rates/Balmer_transient_robustness_sweep.py`|"Balmer_transient_robustness_sweep.py"|none matched|`sensitivity/robustness/…summary.csv`; `figures/` 4 figs|05-10 04:59|
|18|`src/rates/Balmer_transient_ratio.py`|"Balmer_transient_ratio_fast.py"|`cr_matrix/{L_grid,S_grid}.npy`; `Radiative/radiative_rates.csv`|`sensitivity/Balmer_transient_ratio_{timeseries_DTe_{slug},summary}.csv`; `figures/` 3 figs|05-10 04:48|
|19|`src/rates/Balmer_transient_ratio_v3_before_paper_label_patch.py`|"Balmer_transient_ratio_fast.py"|identical to #18|identical to #18|05-10 04:48|
|20|`src/rates/backup_scripts/Balmer_transient_ratio_old_20260510_040033.py`|"Balmer_transient_ratio_fast.py"|identical to #18|identical to #18|05-10 04:00|
|21|`src/rates/Balmer_ratio_sensitivity.py`|"Balmer_ratio_sensitivity_v2.py"|`cr_matrix/{L_grid,S_grid}.npy`; `radiative_rates.csv`|`figures/` 2 stems; `sensitivity/Balmer_ratio_sensitivity_v2_summary.csv`|05-10 03:26|
|22|`src/rates/Halpha_sensitivity.py`|"Halpha_sensitivity.py"|`cr_matrix/{L_grid,S_grid}.npy`|`figures/Halpha_sensitivity_map.*`, `{stem}.*`|05-10 02:53|
|23|`src/rates/S_criterion_fixed.py`|"S_criterion_fixed.py"|`cr_matrix/{L_grid,S_grid}.npy`|`figures/S_criterion_fixed_map.*`; `sensitivity/{S_grid_fixed,a_grid_fixed,p_max_grid_fixed}.npy`|05-10 02:46|
|24|`src/rates/S_criterion.py`|"S_criterion.py"|**two functions each redundantly load** `{L_grid,S_grid}.npy` (lines 69–70 and 131–132)|`figures/S_criterion_{map,collapse}.*`; `sensitivity/{S_grid,p_max_grid}.npy`|05-10 02:29|
|25|`src/rates/regenerate_figures.py`|"regenerate_figures.py"|`mori_zwanzig/{tau_K_grid,tau_relax_MZ,eigenvalues_FF,K_t_ITER_ref,t_grid_ITER_ref}.npy`; `L_grid.npy`|`figures/mz_fig{1,3}_*.*`|04-12 03:28|
|26|`src/rates/check_modal_weights.py`|"check_modal_weights.py"|`L_grid.npy`|**none**|04-12 03:08|
|27|`src/rates/diagnostic.py`|(no docstring)|`mori_zwanzig/{mode_amplitudes,eigenvalues_FF}.npy` (CWD-relative literal)|**none**|04-12 03:06|
|28|`src/rates/verify_partition.py`|"verify_partition.py  (v2 — correct diagnostics)"|`{L_grid,S_grid}.npy`|**none**|04-12 02:54|
|29|`src/rates/compute_term1_fraction.py`|"compute_term1_fraction.py  (v2 — correct diagnostic)"|`{L_grid,S_grid}.npy`; `mori_zwanzig/tau_K_grid.npy`; `validation/tau_QSS_grid.npy` (via `_MZ_DIR/"../../../validation/…"`)|`figures/mz_fig7_term1_fraction.*`|04-12 02:06|
|30|`src/rates/compute_mz_decomposition.py`|"compute_mz_decomposition.py  (v2 — corrected steady-state solve)"|`{L_grid,S_grid}.npy`|`mori_zwanzig/mz_decomp_{t,full,termI,termII}.npy`; `figures/`|04-12 01:13|
|31|`src/rates/check_mz.py`|(no docstring)|indirect via `from assemble_cr_matrix import …`|**none**|04-12 00:16|
|32|`src/rates/qsscheck.py`|(no docstring)|`validation/{tau_relax_grid,tau_QSS_grid,M_grid}.npy`; **grids hardcoded** `np.logspace(0,1,50)`, `np.logspace(12,15,8)`|**none**|04-12 00:04|
|33|`src/rates/check.py`|(no docstring)|`validation/{M_grid,tau_QSS_grid,tau_relax_grid}.npy`; same hardcoded grids; hardcodes `"Thesis M = 611"`|**none**|04-11 23:26|
|34|`src/rates/mori_zwanzig_weekc.py`|"mori_zwanzig_weekC.py"|10 arrays from `mori_zwanzig/`; `validation/tau_QSS_grid.npy`, conditionally `M_grid.npy` else `tau_relax_grid.npy`|`figures/{name}.*`; `mori_zwanzig/weekC_summary.txt`|04-11 21:48|
|35|`src/rates/mori_zwanzing_weekb.py`|"mori_zwanzig_weekB.py"|`mori_zwanzig/{Omega_QSS_grid,tau_relax_MZ}.npy`; `tau_QSS_path` candidate list (`allow_pickle=True`)|`mori_zwanzig/` 9 arrays|04-11 18:17|
|36|`src/rates/mori_zwanzig.py`|"mori_zwanzig.py"|indirect via `assemble_cr_matrix`|`mori_zwanzig/` 10 arrays|04-11 16:35|
|37|`src/analysis/trapping_analysis.py`|"trapping_analysis.py"|indirect via `assemble_cr_matrix`, `escape_factor`|`data/processed/trapping/summary_trapping.txt`|04-08 02:15|
|38|`src/analysis/escape_factor.py`|"escape_factor.py"|**none** (pure quadrature module)|**none**|04-08 01:25|
|39|`src/rates/assemble_cr_matrix.py`|"assemble_cr_matrix.py"|`PATHS`: 13 entries — K_exc_full, K_deexc_full, K_ion_final, alpha_RR_{res,bund}, alpha_3BR_{res,bund}, A_{resolved,bund_res,bund_bund}, gamma_{resolved,bundled}, K_lmix|`cr_matrix/{L_grid,S_grid,Te_grid_L,ne_grid_L}.npy`, `L_meta.csv`|03-28 21:15|
|40|`src/analysis/plot_results.py`|"plot_results.py"|`cr_matrix/{Te_grid_L,ne_grid_L,L_grid,S_grid}.npy`; `validation/{M_grid,tau_QSS_grid,tau_relax_grid}.npy`, `breakdown_map.csv`, `epsilon_traces.npz`|`figures/fig1…fig7`|03-23 19:52|
|41|`src/analysis/unified_scaling.py`|"Validate a reduced QSS-breakdown model against full transient CR truth."|`PATHS`: L_grid, S_grid, Te_grid, ne_grid|`validation/unified_scaling_validation/` figs|03-23 18:12|
|42|`src/analysis/test_scaling.py`|"test_scaling.py"|same 4|`validation/scaling_tests/step{1..4}_*.png`|03-23 17:55|
|43|`src/analysis/physics_tests.py`|"physics_tests.py"|same 4|`validation/physics_tests/` 3 figs|03-23 17:39|
|44|`src/rates/compute_K_VS.py`|"compute_K_VS.py"|`Radiative/H_A_E1_LS_n1_15_physical.csv`|`collisions/vs/{K_VS_exc_table,K_VS_deexc_table,Te_grid_VS}.npy`, `K_VS_metadata.csv`|03-22 20:53|
|45|`src/rates/prepare_adas.py`|(no docstring)|`adas/{scd96_h_long,acd96_h_long}.csv`|`adas/{SCD96,ACD96}_interpolated.csv`|03-22 18:34|
|46|`src/rates/solve_cr.py`|"solve_cr.py"|L_grid, S_grid, Te_grid, ne_grid, K_ion|**none**|03-22 06:34|
|47|`src/rates/pre_assembly_check.py`|"pre_assembly_check.py"|21-entry `PATHS` across K_exc_full/, tics/, recombination/, Radiative/|**none**|03-22 04:28|
|48|`src/rates/recombination_rates.py`|"recombination_rates.py"|`tics/{K_ion_resolved,K_ion_n9_bundled,K_ion_final}.npy`|`recombination/` 5 arrays + `recombination_meta.csv`|03-22 02:19|
|49|`src/rates/radiative_rates.py`|"radiative_rates.py"|`Radiative/H_A_E1_LS_n1_15_physical.csv` (+ `/mnt/user-data/uploads/` fallback)|`Radiative/{A_resolved,A_bund_res,A_bund_bund,gamma_resolved,gamma_bundled}.npy`, `radiative_rates.csv`, index csv|03-22 01:31|
|50|`src/parsers/parse_ccc.py`|"To run: python src/parsers/parse_ccc.py data/raw/ccc/e-H_XSEC_LS data/processed/collisions/ccc"|files under `sys.argv[1]`|`{argv[2]}/ccc_crosssections.{h5,csv}`|03-22 01:15|
|51|`src/rates/ionization_rates.py`|"ionization_rates.py"|`tics/{K_ion_resolved,K_ion_n9_bundled}.npy`|`tics/K_ion_final.npy`, `K_ion_final_meta.csv`, `K_ion_final.csv`|03-22 00:39|
|52|`src/rates/assemble_K_exc.py`|"assemble_K_exc.py"|`ccc/` 4 files; `vs/` 3 files|`K_exc_full/` 5 arrays + `K_exc_meta.csv`, `state_index.csv`|03-17 22:03|
|53|`src/rates/compute_K_CCC.py`|"compute_K_CCC.py"|`ccc/ccc_crosssections.csv`|`ccc/` 4 arrays + metadata; `figures/week2/K_CCC_diagnostic.png`|03-15 03:06|
|54|`src/rates/compute_K_TICS.py`|"compute_K_TICS.py"|`tics/tics_crosssections.csv`|`tics/` 4 arrays + 2 csv|03-14 18:32|
|55|`src/parsers/parse_tics.py`|"parse_tics.py"|files under `data/raw/ccc/e-H_XSEC_LS`|`tics/tics_crosssections.csv`|03-14 15:31|
|56|`src/validation/anderson_benchmark_qc.py`|"Anderson (2000) Benchmark QC — Check 5"|`ccc/ccc_crosssections.csv` (+uploads fallback); `ccc/K_CCC_deexc_table.npy`, metadata, `Te_grid.npy`|`collisions/anderson_benchmark_full.csv`; `figures/anderson_benchmark_full.png`|03-14 01:11|
|57|`src/load_k_ccc.py`|(no docstring)|`ccc/` 4 files|**none**|03-13 23:37|
|58|`src/rates/__init__.py`|(empty)|—|—|03-13 21:14|
|59|`src/parsers/qc2_ccc.py`|"CCC Data Quality Control"|`sys.argv[1]`|`figures/week2/ccc_qc_detailed_balance.png`|03-13 00:15|
|60|`src/parsers/__init__.py`|(empty)|—|—|02-22 17:07|
|61|`src/config/paths.py`|"Path Configuration for Non-Markovian CR Project"|none (dicts only)|**none**|02-22 17:07|
|62|`src/config/__init__.py`|(empty)|—|—|02-22 17:07|
|63|`src/__init__.py`|(empty)|—|—|02-22 17:07|
|64|`src/week2_timescale_map.py`|"Week 2 - Task 2.1"|`data/raw/adas/{scd96_h,acd96_h}.dat` via `ADASRateInterpolator`|`results/week2_tau_map_table.csv`; `figures/week2/week2_{tau,xinf}_heatmap.png`|02-21 15:36|
|65|`src/adas_interpolator.py`|"Log-log interpolator for ADAS ADF11 rate coefficients."|caller-supplied `filepath`|**none**|02-21 03:48|
|66|`src/parser_adasf11.py`|"Extract floats from a line; supports Fortran D exponents."|`open(filepath)`; **two separate `__main__` guards, both execute**|`data/processed/adas/{scd96,acd96}_h_long.csv`|02-21 03:35|

---

## ITEM 2 — WRITE COLLISIONS

The three flagged in the request — **all confirmed**:

| Output path | Writers |
|---|---|
| `validation/M_grid.npy` | `src/validation/qss_analysis.py:411`; `src/validation/validate_gates.py:462` |
| `validation/tau_QSS_grid.npy` | `qss_analysis.py:412`; `validate_gates.py:463` |
| `validation/tau_relax_grid.npy` | `qss_analysis.py:413`; `validate_gates.py:464` |

Both producers compute the spectrum independently. `qss_analysis.py:137` uses
`neg = eigs[eigs < 0.0]`; `validate_gates.py` Gate E (~line 397) uses
`eigs_neg = eigs[eigs < 0.0]`. Same cutoff in the copies read, but two separate
implementations writing one path — last runner wins.

`validation/M_grid.npy` mtime is **2026-08-23 11:17:01**, identical to
`gate_B/C/D.csv` and `gate_summary.txt`, i.e. `validate_gates.py` ran last and
clobbered `qss_analysis.py`'s copies. `qss_analysis.py`'s uniquely-named outputs
(`breakdown_map.csv`, `qss_analysis_summary.txt`) remain at 2026-08-22 14:28:58.

### Other exact-path collisions

| Output path | Writers |
|---|---|
| `data/processed/sensitivity/Balmer_transient_ratio_summary.csv` | `Balmer_transient_ratio.py:734`; `…v3_before_paper_label_patch.py:734`; `backup_scripts/…old_20260510_040033.py:704` |
| `…/Balmer_transient_ratio_timeseries_DTe_{slug}.csv` | same three: 725 / 725 / 695 |
| `figures/Balmer_transient_ratio_DTe_{slug}.{ext}` | same three: 615 / 615 / 585 |
| `figures/Balmer_transient_absolute_errors_DTe_{slug}.{ext}` | same three: 639 / 639 / 609 |
| `figures/Balmer_transient_ratio_errors_overlay.{ext}` | same three: 665 / 665 / 635 |

All three `Balmer_transient_ratio*` files carry identical
`DATA_CR`/`DATA_RAD`/`OUT_DIR`/`FIG_DIR` definitions — including the one under
`backup_scripts/`, which will overwrite live outputs if run.

### Basename-level near-collisions (NOT true collisions)

| Basename | Note |
|---|---|
| `L_grid.npy`, `S_grid.npy`, `Te_grid_L.npy`, `ne_grid_L.npy` | one writer (`assemble_cr_matrix.py`), ~20+ readers |
| `S_grid.npy` | exists in **two** directories: `data/processed/cr_matrix/` (from `assemble_cr_matrix.py`, 07-21) and `data/processed/sensitivity/` (from `S_criterion.py`, 05-20). Unrelated arrays; a naive basename grep conflates them |
| `plateau_slowmode.{csv,txt}` | one writer, three output dirs on disk: `validation/plateau_slowmode/`, `…_w10/`, `…_w100/` — same script invoked >=3x with different `--out` |
| `divertor_map.{csv,txt}` | single writer |

---

## ITEM 3 — DEPENDENCY EDGES

### Producer -> file -> consumers

| Producer | File | Consumers |
|---|---|---|
| `assemble_cr_matrix.py` | `cr_matrix/L_grid.npy` | S_criterion{,_fixed,_3P}, verify_partition, compute_term1_fraction, compute_mz_decomposition, regenerate_figures, check_modal_weights, Halpha_sensitivity, Balmer_ratio_sensitivity, Balmer_transient_ratio (+v3 +backup), test_scaling, unified_scaling, physics_tests, plot_results, solve_cr, qss_analysis, validate_gates, cr_context -> (verify_timescales, verify_ramp_vs_step, verify_boundary_descent, verify_plateau_gridmap, verify_divertor_map, verify_plateau_slowmode) |
| `assemble_cr_matrix.py` | `cr_matrix/S_grid.npy` | same minus check_modal_weights; the 4 verify_* scripts load it explicitly since `CRContext` does not carry S_grid |
| `assemble_cr_matrix.py` | `Te_grid_L.npy`, `ne_grid_L.npy` | plot_results, test_scaling, unified_scaling, physics_tests, solve_cr, qss_analysis, validate_gates, verify_bundling_psm20, cr_context |
| `assemble_cr_matrix.py` | `cr_matrix/L_meta.csv` | **no consumer in src/** |
| `assemble_K_exc.py` | `K_exc_full/{K_exc_full,K_deexc_full}.npy` | assemble_cr_matrix, pre_assembly_check, validate_gates |
| `assemble_K_exc.py` | `K_exc_full/state_index.csv` | pre_assembly_check, cr_context |
| `assemble_K_exc.py` | `K_exc_full/K_exc_meta.csv` | pre_assembly_check, validate_gates |
| `compute_K_CCC.py` | `ccc/{K_CCC_exc_table,K_CCC_deexc_table,Te_grid}.npy`, metadata | assemble_K_exc, load_k_ccc, anderson_benchmark_qc |
| `compute_K_CCC.py` | `ccc/K_exc_to_n10_bundled.npy` | assemble_K_exc |
| `compute_K_VS.py` | `vs/K_VS_*` | assemble_K_exc |
| `compute_K_TICS.py` | `tics/{K_ion_resolved,K_ion_n9_bundled}.npy` | ionization_rates, recombination_rates, pre_assembly_check |
| `ionization_rates.py` | `tics/K_ion_final.npy` | assemble_cr_matrix, pre_assembly_check, solve_cr, validate_gates, recombination_rates |
| `recombination_rates.py` | `recombination/alpha_*` | assemble_cr_matrix, pre_assembly_check |
| `radiative_rates.py` | `Radiative/A_*`, `gamma_*` | assemble_cr_matrix, pre_assembly_check |
| `radiative_rates.py` | `Radiative/radiative_rates.csv` | Balmer_transient_ratio (+v3 +backup), Balmer_ratio_sensitivity |
| `compute_lmix.py` | `lmix/K_lmix.npy` | assemble_cr_matrix, verify_bundling_psm20 |
| `parse_tics.py` | `tics/tics_crosssections.csv` | compute_K_TICS |
| `parse_ccc.py` | `ccc/ccc_crosssections.csv` | compute_K_CCC, qc_ccc, qc2_ccc, anderson_benchmark_qc |
| `parser_adasf11.py` | `adas/{scd96,acd96}_h_long.csv` | prepare_adas |
| `prepare_adas.py` | `adas/{SCD96,ACD96}_interpolated.csv` | validate_gates |
| `qss_analysis.py` **or** `validate_gates.py` | `validation/{M_grid,tau_QSS_grid,tau_relax_grid}.npy` | plot_results, mori_zwanzig_weekc, check, qsscheck, compute_term1_fraction (tau_QSS only) |
| `mori_zwanzig.py` | `mori_zwanzig/` 10 arrays | regenerate_figures, mori_zwanzig_weekc, mori_zwanzing_weekb |
| `mori_zwanzing_weekb.py` | `mori_zwanzig/{tau_K_grid,M_MZ_grid,K_tilde_0_grid,Omega_ratio_grid,mode_amplitudes}.npy` | mori_zwanzig_weekc, regenerate_figures, compute_term1_fraction |
| `Balmer_transient_regime_heatmap.py` | `sensitivity/regime/…DTe_p0p60.csv` | Balmer_timescale_audit, plot_regime_persistence_fixed |

### Loaded but produced by no script in src/ (external / manual inputs)

| File | Loaded by |
|---|---|
| `data/raw/ccc/e-H_XSEC_LS/*` | parse_ccc, parse_tics |
| `data/raw/adas/{scd96_h,acd96_h}.dat` | parser_adasf11, week2_timescale_map / adas_interpolator |
| `data/processed/Radiative/H_A_E1_LS_n1_15_physical.csv` | radiative_rates, compute_K_VS — mtime 2026-02-13, manually placed |
| `/mnt/user-data/uploads/…` (2 paths) | anderson_benchmark_qc, radiative_rates — sandbox fallback candidates |

### Freshness flags (consumer mtime OLDER than the file it loads)

| File | File mtime | Consumers older than it | Flag |
|---|---|---|---|
| `cr_matrix/{L_grid,S_grid,Te_grid_L,ne_grid_L}.npy` | 2026-07-21 20:44 | all of `src/rates/` inventory rows 12–24 (S_criterion_3P 05-20 … S_criterion 05-10), verify_bundling_psm20 (05-20) | **FLAGGED** — edited before the 21 Jul regeneration |
| same | same | the 08-22 / 08-23 validation scripts | not flagged |
| `validation/{M_grid,tau_QSS_grid,tau_relax_grid}.npy` | 2026-08-23 11:17 | compute_term1_fraction (04-12), mori_zwanzig_weekc (04-11), check / qsscheck (04-11/12) | **FLAGGED** — 4 months older than the file they read |
| `sensitivity/Balmer_transient_ratio_summary.csv` | 2026-05-10 04:48 | — | output predates the 21 Jul L_grid regeneration; not re-run since |
| `mori_zwanzig/*` | 2026-04-11 16:35 | consumers all 04-12 | not flagged; but the MZ chain rebuilds L from March-dated rate tables, so it does not include the post-Jul l-mixing fix unless re-run |

No consumer referencing a missing on-disk file was found, except the two
`/mnt/user-data/uploads/` fallbacks (existence not checked — plainly external).

---

## ITEM 4 — WHAT DEPENDS ON L_grid.npy

```
find /Users/phi/Desktop/non_markovian_cr -name "L_grid.npy" -type f
-> data/processed/cr_matrix/L_grid.npy   (only copy)
stat -f '%Sm %N' -t '%Y-%m-%d %H:%M:%S' -> 2026-07-21 20:44:53
```

Matches the request's reference (2026-07-21 20:44) to the minute.
**CLAUDE.md states "L_grid.npy regenerated 14 Jul 2026" — the disk says 21 Jul,
seven days later.** Disk used as the comparison basis throughout.

### Level 0 — direct consumers

| script | depth | loads | writes |
|---|---|---|---|
| `rates/assemble_cr_matrix.py:325` | **producer** | raw rate tables | `L_grid.npy`, `S_grid.npy`, `Te_grid_L.npy`, `ne_grid_L.npy`, `L_meta.csv` |
| `validation/cr_context.py:171` | 0 | L_grid + Te/ne grids + state_index.csv | none (in-memory `CRContext`) |
| `rates/solve_cr.py:130` | 0 | L_grid, S_grid | none |
| `rates/check_modal_weights.py:24` | 0 | L_grid | none |
| `rates/verify_partition.py:42` | 0 | L_grid, S_grid | none |
| `rates/compute_mz_decomposition.py:115` | 0 | L_grid | `mori_zwanzig/mz_decomp_*.npy`, figures |
| `rates/compute_term1_fraction.py:100` | 0 **and 2** | L_grid, S_grid, `mori_zwanzig/tau_K_grid.npy`, `validation/tau_QSS_grid.npy` | `figures/mz_fig7_*` |
| `rates/regenerate_figures.py:43` | 0 **and 1** | L_grid + 5 `mori_zwanzig/*` arrays | `figures/mz_fig1_*`, `mz_fig3_*` |
| `rates/S_criterion.py:69,131` | 0 | L_grid | `sensitivity/{S_grid,p_max_grid}.npy`, figures |
| `rates/S_criterion_fixed.py:112` | 0 | L_grid | `sensitivity/{S_grid_fixed,a_grid_fixed,p_max_grid_fixed}.npy`, `S_criterion_fixed_results.npz`, figures |
| `rates/S_criterion_3P.py:91` | 0 | L_grid | `sensitivity/{S_3P_grid,a_3P_grid}.npy`, `eps_3P_results.npz`, figures |
| `rates/Halpha_sensitivity.py:138` | 0 | L_grid | `sensitivity/Halpha_sensitivity_results.npz`, figures |
| `rates/Balmer_ratio_sensitivity.py:171` | 0 | L_grid | `sensitivity/Balmer_ratio_sensitivity_v2_*`, figures |
| `rates/Balmer_transient_ratio.py:194,202` | 0 | L_grid, S_grid | `sensitivity/Balmer_transient_ratio_*`, figures |
| `rates/Balmer_transient_regime_heatmap.py:337` | 0 (via `btr.load_cr_grids()`) | L_grid, S_grid | `sensitivity/regime/*.{csv,tex}`, `figures/regime/*` |
| `rates/Balmer_transient_robustness_sweep.py:348` | 0 (via btr) | L_grid, S_grid | `sensitivity/robustness/*` |
| `rates/backup_scripts/Balmer_transient_ratio_old_….py:185,193` | 0 | L_grid, S_grid | **same target paths as the live script** |
| `rates/Balmer_transient_ratio_v3_before_paper_label_patch.py:194,202` | 0 | L_grid, S_grid | **same target paths again** |
| `analysis/unified_scaling.py:44,138` | 0 | L_grid, S_grid | `validation/unified_scaling_validation/*.png` |
| `analysis/physics_tests.py:39,75` | 0 | L_grid, S_grid | `validation/physics_tests/*.png`, `report.md` |
| `analysis/test_scaling.py:44,82` | 0 | L_grid, S_grid | `validation/scaling_tests/*.png`, `report.md` |
| `analysis/plot_results.py:333` | 0 **and 1** | L_grid + `validation/{M,tau_QSS,tau_relax}_grid.npy`, `breakdown_map.csv` | 7 figures |
| `validation/validate_gates.py:62` | 0 | L_grid, S_grid, Te/ne grids, K_ion, K_exc, K_deexc | `gate_{B,C,D}.csv`, `gate_summary.txt`, **M_grid.npy, tau_QSS_grid.npy, tau_relax_grid.npy** |
| `validation/qss_analysis.py:74` | 0 | L_grid, S_grid, Te/ne grids | `breakdown_map.csv`, `qss_analysis_summary.txt`, **same three .npy** |
| `validation/verify_timescales.py` | 0 (via cr_context) | L_grid | `validation/{spectrum_testpoint,timescale_verification}.csv` |
| `validation/verify_ramp_vs_step.py` | 0 (via cr_context) | L_grid | `validation/ramp_vs_step.csv` |
| `validation/verify_boundary_descent.py` | 0 (via cr_context) | L_grid | `validation/boundary_descent.csv` |
| `validation/verify_plateau_slowmode.py:139` | 0 (via cr_context) | L_grid | `validation/plateau_slowmode{,_w10,_w100}/*` |
| `validation/verify_plateau_gridmap.py:115` | 0 (via cr_context) | L_grid | `validation/plateau_gridmap/*` |
| `validation/verify_divertor_map.py:116` | 0 (via cr_context) | L_grid | `validation/divertor_map/*` |

### Level 1–2

| script | depth | loads | writes |
|---|---|---|---|
| `analysis/plot_results.py:79-82` | 1 | `validation/{M,tau_QSS,tau_relax}_grid.npy`, `breakdown_map.csv` | figures |
| `rates/check.py:2-4` | 1 | the three validation `.npy` | none |
| `rates/qsscheck.py:3-5` | 1 | same three | none |
| `rates/compute_term1_fraction.py:103` | 2 | `validation/tau_QSS_grid.npy` (via `_MZ_DIR/../../../`) | figures |
| `rates/mori_zwanzig_weekc.py:107-110` | 2 | `validation/{tau_QSS,M}_grid.npy` | figures, `weekC_summary.txt` |
| `rates/mori_zwanzing_weekb.py:239-245` | 2 (guarded) | `validation/tau_QSS_grid.npy` | `mori_zwanzig/*` |
| `rates/Balmer_timescale_audit.py:221` | 1 | `sensitivity/regime/…DTe_p0p60.csv` | audit csv/tex/txt, figures |

### Downstream output files vs L_grid.npy (2026-07-21 20:44:53)

| file | mtime | vs L_grid |
|---|---|---|
| `mori_zwanzig/mz_decomp_{t,full,termI,termII}.npy` | 04-12 01:13:37 | **OLDER** |
| `sensitivity/{S_grid_fixed,a_grid_fixed,p_max_grid_fixed}.npy`, `S_criterion_fixed_results.npz` | 05-10 02:46:08 | **OLDER** |
| `sensitivity/S_grid.npy`, `p_max_grid.npy` | 05-20 23:04:26 | **OLDER** |
| `sensitivity/{S_3P_grid,a_3P_grid}.npy`, `eps_3P_results.npz` | 05-20 23:28:16 | **OLDER** |
| `sensitivity/Halpha_sensitivity_results.npz` | 05-10 02:53:11 | **OLDER** |
| `sensitivity/Balmer_ratio_sensitivity_v2_{results.npz,summary.csv}` | 05-10 03:26:42 | **OLDER** |
| `sensitivity/Balmer_transient_ratio_summary.csv` | 05-10 04:48:36 | **OLDER** |
| `cr_matrix/{S_grid,Te_grid_L,ne_grid_L}.npy` | 07-21 20:44:53 | same run |
| `validation/timescale_verification.csv`, `boundary_descent.csv` | 08-06 02:46:32 | NEWER |
| `validation/spectrum_testpoint.csv` | 08-22 11:34:19 | NEWER |
| `validation/ramp_vs_step.csv` | 08-22 12:25:15 | NEWER |
| `validation/breakdown_map.csv`, `qss_analysis_summary.txt` | 08-22 14:28:58 | NEWER |
| `validation/plateau_slowmode/*` | 08-23 00:16:02 | NEWER |
| `validation/plateau_gridmap/*` | 08-23 00:22:30 | NEWER |
| `validation/plateau_slowmode_w10/`, `_w100/` | 08-23 11:15:18 | NEWER |
| `validation/{M,tau_QSS,tau_relax}_grid.npy`, `gate_{B,C,D}.csv`, `gate_summary.txt` | 08-23 11:17:01 | NEWER |
| `validation/divertor_map/*` | 08-23 11:20:01 | NEWER |

Nothing in the closure is MISSING on disk.

### Two structural findings

1. **The write collision is live and currently resolved in one direction.**
   `validation/{M,tau_QSS,tau_relax}_grid.npy` carry mtime `08-23 11:17:01`,
   byte-identical to `gate_B/C/D.csv` and `gate_summary.txt`. `validate_gates.py`
   ran last. The six downstream readers cannot tell which producer's numbers
   they hold.

2. **A parallel branch that never reads the file.** `rates/mori_zwanzig.py:82`
   does `from assemble_cr_matrix import load_rates, TE_GRID, NE_GRID, build_L`
   and rebuilds L in memory from raw tables. Its outputs are dated
   `04-11 16:35–18:17` — before even the 14 Jul date CLAUDE.md claims. Yet
   `regenerate_figures.py`, `mori_zwanzig_weekc.py` and `compute_term1_fraction.py`
   load those 3-month-stale arrays **in the same script as a fresh `L_grid.npy`**.

---

## ITEM 5 — THE BALMER OBSERVABLE

| File | Size | Lines | mtime |
|---|---|---|---|
| `src/rates/Balmer_ratio_sensitivity.py` | 24784 | 711 | 2026-05-10 03:26:41 |
| `src/rates/Balmer_transient_ratio.py` | 28023 | 787 | 2026-05-10 04:48:20 |

### All Balmer* files under src/

| Path | mtime | Status |
|---|---|---|
| `src/rates/Balmer_timescale_audit.py` | 05-10 05:36:59 | consumer (`import Balmer_transient_ratio as btr`) |
| `src/rates/Balmer_transient_regime_heatmap.py` | 05-10 05:19:46 | consumer |
| `src/rates/Balmer_transient_robustness_sweep.py` | 05-10 04:59:43 | consumer |
| `src/rates/Balmer_transient_ratio_v3_before_paper_label_patch.py` | 05-10 04:48:01 | **`diff` = 0 lines vs the current file — byte-identical despite the "before patch" name** |
| `src/rates/backup_scripts/Balmer_transient_ratio_old_20260510_040033.py` | 05-10 04:00:33 | **103 lines differ, concentrated in the A-coefficient lookup** |
| `src/rates/__pycache__/Balmer_transient_ratio.cpython-311.pyc` | 05-10 04:59:43 | bytecode, not decompiled |

The backup selects radiative rows by `idx_upper`/`idx_lower` alone and **sums**
on multiple matches (`A_val = float(rows["A_s-1"].sum())`, warning only). The
current file filters on `type == "res_to_res"` plus the full `(n,l)->(n,l)`
quadruple and raises unless exactly one row matches.

The collision is real in `data/processed/Radiative/radiative_rates.csv`
(414 rows). For `(idx_upper=3, idx_lower=2)` = 3S->2P:

```
bund_to_bund,12,-1,n12(bund),11,-1,n11(bund),28004.78976388889,3,2
bund_to_res,12,-1,n12(bund),2,1,2P,20943.965277777777,3,2
res_to_res,3,0,3S,2,1,2P,6317200.0,3,2
```

File order puts `bund_to_bund` first (line 7), `res_to_res` last (line 277), for
all four colliding pairs checked. **No script on disk takes `rows.iloc[0]`
unconditionally.** The -46% Halpha artifact is NOT attributed to this — neither
target file nor the backup exhibits that bug.

### Balmer_ratio_sensitivity.py

#### (a) weight construction

```python
122	@dataclass
123	class LineChannel:
124	    line_name: str
125	    label: str
126	    n_upper: int
127	    l_upper: int
128	    n_lower: int
129	    l_lower: int
130	    idx_upper: int
131	    idx_lower: int
132	    A_s: float
133	    weight: float
```

```python
284	def photon_energy_weight(line_name: str) -> float:
285	    if not USE_PHOTON_ENERGY:
286	        return 1.0
287	    # h*c/lambda constant is unnecessary; only relative line factors matter.
288	    return 1.0 / LINE_WAVELENGTH_NM[line_name]
289	
290	
291	def lookup_channel(df: pd.DataFrame, line_name: str, spec: Tuple[int, int, int, int, str]) -> LineChannel:
292	    n_u, l_u, n_l, l_l, label = spec
293	    rows = df[
294	        (df["n_upper"] == n_u)
295	        & (df["l_upper"] == l_u)
296	        & (df["n_lower"] == n_l)
297	        & (df["l_lower"] == l_l)
298	    ]
299	    if len(rows) != 1:
300	        raise ValueError(
301	            f"Expected exactly one radiative row for {label} "
302	            f"({n_u},{l_u})->({n_l},{l_l}); found {len(rows)}."
303	        )
304	    row = rows.iloc[0]
305	    A_s = float(row["A_s-1"])
306	    energy_factor = photon_energy_weight(line_name)
307	    return LineChannel(
308	        line_name=line_name,
309	        label=label,
310	        n_upper=n_u,
311	        l_upper=l_u,
312	        n_lower=n_l,
313	        l_lower=l_l,
314	        idx_upper=int(row["idx_upper"]),
315	        idx_lower=int(row["idx_lower"]),
316	        A_s=A_s,
317	        weight=A_s * energy_factor,
318	    )
319	
320	
321	def load_line_channels() -> Dict[str, List[LineChannel]]:
322	    df = load_radiative_rates_csv()
323	    channels: Dict[str, List[LineChannel]] = {}
324	    for line_name, specs in LINE_CHANNELS.items():
325	        channels[line_name] = [lookup_channel(df, line_name, spec) for spec in specs]
326	    return channels
```

The literal names `weight_Halpha_*` / `weight_Hbeta_*` exist only as output keys:

```python
630	    for line_name, chs in channels.items():
631	        for ch in chs:
632	            save_dict[f"A_{line_name}_{ch.label}"] = np.asarray(ch.A_s)
633	            save_dict[f"weight_{line_name}_{ch.label}"] = np.asarray(ch.weight)
634	            save_dict[f"idx_upper_{line_name}_{ch.label}"] = np.asarray(ch.idx_upper)
635	            save_dict[f"idx_lower_{line_name}_{ch.label}"] = np.asarray(ch.idx_lower)
```

This file does **not** filter on `type == "res_to_res"`. It is protected only by
the `len(rows) != 1` raise, and structurally by the fact that bundled rows carry
`l = -1`, so an `(n,l)`-quadruple filter cannot match them.

#### (b) l-resolved populations — solve, not statistical

`grep` for `(2l+1)`, `statistical`, `stat_weight`, `degenerac*`: **zero hits.**

```python
195	def steady_state(L: np.ndarray, S_src: np.ndarray, n_ion: float = N_ION) -> np.ndarray:
196	    """Solve L n_ss = -S_src * n_ion."""
197	    n_ss = np.linalg.solve(L, -S_src * n_ion)
198	    if not np.all(np.isfinite(n_ss)):
199	        raise FloatingPointError("Non-finite steady-state population.")
200	
201	    # Clip numerical roundoff. Warn only for meaningful negativity.
202	    scale = max(float(np.max(np.abs(n_ss))), 1.0)
203	    tol = 1e-10 * scale
204	    min_val = float(np.min(n_ss))
205	    if min_val < -tol:
206	        print(
207	            f"WARNING: significant negative steady-state population: "
208	            f"min={min_val:.3e}, tol={tol:.3e}. Clipping for diagnostic."
209	        )
210	    return np.where(n_ss < 0.0, 0.0, n_ss)
```

Extraction is direct array indexing, no redistribution:

```python
351	def line_emissivity_proxy(n_ss: np.ndarray, line_name: str, channels: Dict[str, List[LineChannel]]) -> float:
352	    total = 0.0
353	    for ch in channels[line_name]:
354	        total += ch.weight * n_ss[ch.idx_upper]
355	    return float(total)
```

#### (c) I_Halpha / I_Hbeta

```python
358	def compute_observable_grids(
359	    L_grid: np.ndarray,
360	    S_grid_src: np.ndarray,
361	    channels: Dict[str, List[LineChannel]],
362	) -> Dict[str, np.ndarray]:
363	    n_te, n_ne, _, _ = L_grid.shape
364	    I_Ha = np.full((n_te, n_ne), np.nan, dtype=float)
365	    I_Hb = np.full((n_te, n_ne), np.nan, dtype=float)
366	    R_ab = np.full((n_te, n_ne), np.nan, dtype=float)
367	
368	    for ti in range(n_te):
369	        for ni in range(n_ne):
370	            n_ss = steady_state(L_grid[ti, ni], S_grid_src[ti, ni])
371	            ha = line_emissivity_proxy(n_ss, "Halpha", channels)
372	            hb = line_emissivity_proxy(n_ss, "Hbeta", channels)
373	            I_Ha[ti, ni] = ha
374	            I_Hb[ti, ni] = hb
375	            if np.isfinite(ha) and np.isfinite(hb) and hb > MIN_POSITIVE:
376	                R_ab[ti, ni] = ha / hb
377	
378	    return {
379	        "Halpha": I_Ha,
380	        "Hbeta": I_Hb,
381	        "Halpha_over_Hbeta": R_ab,
382	    }
```

#### (d) reported error

```python
401	def compute_observable_step_results(
402	    delta_te_list: List[float],
403	    sensitivity: ObservableSensitivity,
404	) -> Dict[float, ObservableStepResult]:
405	    n_te, n_ne = sensitivity.O_grid.shape
406	    results: Dict[float, ObservableStepResult] = {}
407	
408	    for delta_nominal in delta_te_list:
409	        eps_actual = np.full((n_te, n_ne), np.nan, dtype=float)
410	        pred_linear = np.full((n_te, n_ne), np.nan, dtype=float)
411	        pred_exp = np.full((n_te, n_ne), np.nan, dtype=float)
412	        actual_delta_te = np.full((n_te, n_ne), np.nan, dtype=float)
413	
414	        for ti in range(n_te):
415	            ti_new, dte_actual = nearest_new_index(ti, delta_nominal)
416	            if ti_new is None:
417	                continue
418	
419	            for ni in range(n_ne):
420	                O_old = sensitivity.O_grid[ti, ni]
421	                O_new = sensitivity.O_grid[ti_new, ni]
422	                if np.isfinite(O_old) and np.isfinite(O_new) and O_new > MIN_POSITIVE:
423	                    eps_actual[ti, ni] = abs(O_old - O_new) / O_new
424	
425	                a = sensitivity.a_grid[ti, ni]
426	                if np.isfinite(a):
427	                    actual_delta_te[ti, ni] = dte_actual
428	                    pred_linear[ti, ni] = abs(a * dte_actual)
429	                    exponent = float(np.clip(-a * dte_actual, -700.0, 700.0))
430	                    pred_exp[ti, ni] = abs(np.exp(exponent) - 1.0)
```

Printed at the benchmark point:

```python
580	    for delta_nominal, res in step_results.items():
581	        eps = res.eps_actual[ti_r, ni_r]
582	        lin = res.pred_linear[ti_r, ni_r]
583	        exp = res.pred_exp[ti_r, ni_r]
584	        dta = res.actual_delta_te[ti_r, ni_r]
585	        ratio_lin = eps / lin if np.isfinite(eps) and np.isfinite(lin) and lin > 0 else np.nan
586	        ratio_exp = eps / exp if np.isfinite(eps) and np.isfinite(exp) and exp > 0 else np.nan
```

**Loads:** `data/processed/cr_matrix/L_grid.npy` (171), `S_grid.npy` (172),
`data/processed/Radiative/radiative_rates.csv` (77 / 266), `TE_GRID`/`NE_GRID`
imported from `assemble_cr_matrix` (80).
**Writes:** `sensitivity/Balmer_ratio_sensitivity_v2_results.npz` (650),
`..._v2_summary.csv` (651), figures `Balmer_v2_{name}_sensitivity_map` (482–484),
`Balmer_v2_{name}_collapse_{linear|exp}` (546–548).

Atomic data source, quoted:

```
5	Balmer observable sensitivity analysis using the actual Hoang-Binh radiative
6	A-values stored in:
7	
8	    data/processed/Radiative/radiative_rates.csv
```

### Balmer_transient_ratio.py

#### (a) weight construction — with hardcoded state indices

```python
109	N_ION_DEFAULT = 1e14
110	MIN_POSITIVE = 1e-300
111	
112	# Assumed state indices from your 43-state ordering.
113	IDX_2S = 1
114	IDX_2P = 2
115	IDX_3S = 3
116	IDX_3P = 4
117	IDX_3D = 5
118	IDX_4S = 6
119	IDX_4P = 7
120	IDX_4D = 8
121	
122	# Required radiative channels.
123	#
124	# IMPORTANT:
125	# radiative_rates.csv contains idx_upper/idx_lower collisions for bundled rows.
126	# Therefore, DO NOT identify A-values by idx_upper/idx_lower alone.
127	# Use the physical resolved transition selector (type=res_to_res, n,l -> n,l),
128	# while using the state_index_upper for multiplying the CR population vector.
129	#
130	# Tuple format:
131	#   (state_index_upper, state_index_lower, n_upper, l_upper, n_lower, l_lower, label)
132	LINE_CHANNELS = {
133	    "Halpha": [
134	        (IDX_3S, IDX_2P, 3, 0, 2, 1, "3S_to_2P"),
135	        (IDX_3P, IDX_2S, 3, 1, 2, 0, "3P_to_2S"),
136	        (IDX_3D, IDX_2P, 3, 2, 2, 1, "3D_to_2P"),
137	    ],
138	    "Hbeta": [
139	        (IDX_4S, IDX_2P, 4, 0, 2, 1, "4S_to_2P"),
140	        (IDX_4P, IDX_2S, 4, 1, 2, 0, "4P_to_2S"),
141	        (IDX_4D, IDX_2P, 4, 2, 2, 1, "4D_to_2P"),
142	    ],
143	}
```

```python
221	def load_radiative_weights(use_photon_energy: bool = False) -> LineWeights:
222	    if not DATA_RAD.exists():
223	        raise FileNotFoundError(
224	            f"Missing radiative rates CSV: {DATA_RAD}\n"
225	            "Expected relative path from repo root: data/processed/Radiative/radiative_rates.csv"
226	        )
227	
228	    df = pd.read_csv(DATA_RAD)
229	    required_cols = {
230	        "type", "n_upper", "l_upper", "n_lower", "l_lower",
231	        "label_upper", "label_lower", "A_s-1", "idx_upper", "idx_lower"
232	    }
233	    missing = required_cols.difference(df.columns)
234	    if missing:
235	        raise ValueError(f"radiative_rates.csv missing columns: {sorted(missing)}")
236	
237	    weights: Dict[str, Dict[str, float]] = {}
238	
239	    for line, channels in LINE_CHANNELS.items():
240	        weights[line] = {}
241	        photon_factor = 1.0
242	        if use_photon_energy:
243	            photon_factor = H_PLANCK * C_LIGHT / LINE_WAVELENGTH_M[line]
244	
245	        for state_upper, state_lower, n_u, l_u, n_l, l_l, label in channels:
246	            rows = df[
247	                (df["type"] == "res_to_res")
248	                & (df["n_upper"] == n_u)
249	                & (df["l_upper"] == l_u)
250	                & (df["n_lower"] == n_l)
251	                & (df["l_lower"] == l_l)
252	            ]
253	            if rows.empty:
254	                raise ValueError(
255	                    f"No resolved radiative row found for {label}: "
256	                    f"type=res_to_res, ({n_u},{l_u})->({n_l},{l_l})"
257	                )
258	            if len(rows) != 1:
259	                raise ValueError(
260	                    f"Expected exactly one resolved row for {label}; found {len(rows)}.\n"
261	                    f"Rows:\n{rows.to_string(index=False)}"
262	                )
263	
264	            row = rows.iloc[0]
265	            A_val = float(row["A_s-1"])
266	
267	            # Sanity check: the resolved row should map to the CR state indices.
268	            csv_upper = int(row["idx_upper"])
269	            csv_lower = int(row["idx_lower"])
270	            if csv_upper != state_upper or csv_lower != state_lower:
271	                raise ValueError(
272	                    f"State-index mismatch for {label}: CSV idx=({csv_upper},{csv_lower}), "
273	                    f"expected=({state_upper},{state_lower}). This means state ordering changed."
274	                )
275	
276	            weights[line][label] = A_val * photon_factor
277	
278	    return LineWeights(weights=weights)
```

Lines 113–120 are a re-typed state ordering, not loaded from `cr_context.py`.
The 267–274 check catches drift between the CSV and those constants, but the
physical identity of `IDX_3P = 4` itself is never verified against a canonical
ordering file inside this script.

#### (b) populations — eigen-propagation, not statistical

`grep` for `(2l+1)` / `statistical` / `degenerac*`: **zero hits.**

```python
455	    eigvals, eigvecs = eig(L_new)
456	    cond_v = np.linalg.cond(eigvecs)
457	    if cond_v > 1e12:
458	        print(
459	            f"WARNING: eigenvector matrix is ill-conditioned, cond(V)={cond_v:.3e}. "
460	            "Use this fast result as a diagnostic and cross-check with expm_multiply if needed."
461	        )
462	
463	    coeff = scipy_solve(eigvecs, deviation0, assume_a="gen")
464	    exp_wt = np.exp(np.outer(times, eigvals))
465	    deviations = (exp_wt * coeff[None, :]) @ eigvecs.T
466	    N = n_ss_new[None, :] + np.real_if_close(deviations, tol=1000).real
467	
468	    # Clip tiny negative numerical noise for line diagnostics.
469	    N = np.where(N < 0.0, 0.0, N)
```

```python
332	def line_intensity(n: np.ndarray, line: str, line_weights: LineWeights) -> float:
333	    total = 0.0
334	    for state_upper, _state_lower, _n_u, _l_u, _n_l, _l_l, label in LINE_CHANNELS[line]:
335	        total += line_weights.weights[line][label] * n[state_upper]
336	    return float(total)
```

#### (c) I_Halpha / I_Hbeta

```python
339	def compute_observables(n: np.ndarray, line_weights: LineWeights) -> Tuple[float, float, float]:
340	    Ha = line_intensity(n, "Halpha", line_weights)
341	    Hb = line_intensity(n, "Hbeta", line_weights)
342	    ratio = Ha / Hb if Hb > MIN_POSITIVE else np.nan
343	    return Ha, Hb, ratio
```

```python
439	    Ha_old, Hb_old, R_old = compute_observables(n_ss_old, line_weights)
440	    Ha_new, Hb_new, R_new = compute_observables(n_ss_new, line_weights)
```

```python
473	    Ha_cr = np.empty(n_time_steps, dtype=float)
474	    Hb_cr = np.empty(n_time_steps, dtype=float)
475	    R_cr = np.empty(n_time_steps, dtype=float)
476	
477	    for k in range(n_time_steps):
478	        Ha_cr[k], Hb_cr[k], R_cr[k] = compute_observables(N[k, :], line_weights)
```

#### (d) reported ratio error

```python
382	def safe_fractional_error(actual: np.ndarray | float, target: float) -> np.ndarray:
383	    if not np.isfinite(target) or abs(target) <= MIN_POSITIVE:
384	        return np.full_like(np.asarray(actual, dtype=float), np.nan, dtype=float)
385	    return (np.asarray(actual, dtype=float) - target) / target
```

```python
480	    err_Ha = safe_fractional_error(Ha_cr, Ha_new)
481	    err_Hb = safe_fractional_error(Hb_cr, Hb_new)
482	    err_R = safe_fractional_error(R_cr, R_new)
```

Signed, normalized to the **post-step** QSS value. Summarized via:

```python
388	def threshold_duration(times: np.ndarray, err: np.ndarray, threshold: float) -> float:
389	    """
390	    Duration from t=0 until |err| drops below threshold for good.
391	    Uses the last sampled time with |err| >= threshold.
392	    """
393	    mag = np.abs(err)
394	    valid = np.isfinite(mag)
395	    above = valid & (mag >= threshold)
396	    if not np.any(above):
397	        return 0.0
398	    return float(times[np.where(above)[0][-1]])
399	
400	
401	def peak_error(times: np.ndarray, err: np.ndarray) -> Tuple[float, float]:
402	    mag = np.abs(err)
403	    valid = np.isfinite(mag)
404	    if not np.any(valid):
405	        return np.nan, np.nan
406	    idx_valid = np.where(valid)[0]
407	    idx = idx_valid[int(np.nanargmax(mag[valid]))]
408	    return float(err[idx]), float(times[idx])
```

**Loads:** `cr_matrix/L_grid.npy`, `S_grid.npy` (194–195),
`Radiative/radiative_rates.csv` (97), `TE_GRID`/`NE_GRID` from
`assemble_cr_matrix` (102).
**Writes:** `sensitivity/Balmer_transient_ratio_summary.csv` (734–735),
`..._timeseries_DTe_{slug}.csv` (725–726), figures at 614–615, 638–639, 664–665.

### The two files use different denominators for their headline error

- `Balmer_ratio_sensitivity.py`: `abs(O_old - O_new) / O_new` — unsigned,
  QSS-vs-QSS across a Te step.
- `Balmer_transient_ratio.py`: `(I(t) - I_new) / I_new` — signed,
  transient-vs-post-step-QSS.

---

## ITEM 6 — ERROR METRIC INVENTORY

**The claim of >=3 incompatible `eps_step` definitions is CONFIRMED** — 10
occurrences across 10 files in at least 6 structurally distinct families. Two
files document the conflict themselves.

`src/validation/verify_divertor_map.py:68-71`:

```
68	  - eps_step here is the relative change of ONE scalar observable between two
69	    QSS states. It is NOT the eps_step in qss_analysis.py, which is a
70	    max-over-42-states ratio error normalised by the ground state. Same name,
71	    different quantity -- must be resolved in the Ch. 3 notation.
```

`src/rates/S_criterion_fixed.py:1-12`:

```
7	This script fixes the earlier S_criterion.py issues:
8	
9	1. eps_step is computed using population ratios:
10	       r_p = n_p / n_1s
11	   not absolute populations n_p.
```

### Every eps_step occurrence

| File | Line | Expression | Family |
|---|---|---|---|
| `src/rates/S_criterion.py` | 154–156 | `err[mask] = np.abs(n_old[1:][mask] - n_new[1:][mask]) / n_new[1:][mask]`; `eps[ti,ni] = np.max(err)` | **A** raw population diff, max over states, /new |
| `src/rates/S_criterion_fixed.py` | 381, 393 | `err[valid_r] = np.abs(r_old[1:][valid_r] - r_new[1:][valid_r]) / r_new[1:][valid_r]`, `r = n/n_1s` | **B** ratio-to-ground diff, /new |
| `src/rates/S_criterion_3P.py` | 184 | `eps[ti,ni] = abs(r_old - r_new) / r_old` | **C** single state (3P), **/old** |
| `src/rates/verify_partition.py` | 134 | `eps_step = 1.53 * np.exp(-0.37 * TE_GRID[ti_old]) + 0.01` | **D** hardcoded fit, no data dependence |
| `src/analysis/test_scaling.py` | 165, 183 | `eps_all = np.abs(r0 - r1) / (r1 + 1e-60)`; `eps = float(np.max(eps_all))` — `n1` from **interpolated** `get_ss_interp` | **B**, interpolated |
| `src/analysis/physics_tests.py` | 106, 114 | `return float((np.abs(r0 - r1) / (r1 + 1e-60)).max())` | **B**, interpolated |
| `src/analysis/unified_scaling.py` | 335–339 | `rel_err = (I_true - I_qss)/np.maximum(I_qss,1e-60)`; `eps_step_Ha = float(abs_err[0])` — `I_true` from `solve_ivp` | **F** transient ODE, single line, t=0 sample |
| `src/validation/verify_divertor_map.py` | 200–202 | `Rq = n_new[N3].sum()/n_new[N4].sum()`; `es = abs(n_old[N3].sum()/n_old[N4].sum()/Rq - 1.0)` | **E** shell-sum ratio |
| `src/validation/verify_plateau_slowmode.py` | 260–265, 400 | `d_step = R_of(n_old)/Rq - 1.0`; `eps_step=abs(d_step)` | **E** |
| `src/validation/verify_plateau_gridmap.py` | 205–208, 219 | `d_step = R_old/Rq - 1.0`; `eps_step=abs(d_step)` | **E** |

**Families, plainly:**

- **A** (raw population difference) is explicitly disavowed by
  `S_criterion_fixed.py`'s own docstring as an issue that was "fixed."
- **B** (ratio difference, /new, max over ~42 states) appears in four files, but
  `test_scaling.py` and `physics_tests.py` use interpolated off-grid steady states
  rather than exact grid-snapped indices.
- **C** normalizes to the **old** state — the opposite denominator convention —
  for a single state only.
- **D** is not computed from the CR solve at all.
- **E** is a single shell-sum-ratio observable, not a max-over-states quantity.
- **F** is derived from a full time-dependent ODE integration of a single line
  intensity, sampled at the first output time.

**Second inconsistency inside family D:** `verify_partition.py:134` hardcodes
`1.53*exp(-0.37*Te)+0.01`, while `src/analysis/plot_results.py:459` states the fit
as `1.52*exp(-0.42*Te)+0.04`. Different coefficients, same label. Whether
`plot_results.py`'s comment derives from its own `bd_s['eps_step']` column (a real
computed value, line 469) **cannot be determined statically.**

**Cross-file consumption under a mismatched definition:**
`src/rates/verify_partition.py:144` computes `N_A / eps_step`, comparing a
data-derived `N_A = abs(n_3P_old - n_3P_new)/n_3P_new` (line 123) against the
hardcoded family-D `eps_step`, printing "should be ~1 if N_A = epsilon_step".

### Other error/deviation definitions

| File | Line | Var | Expression | Relative? | Scope |
|---|---|---|---|---|---|
| `src/rates/assemble_cr_matrix.py` | 278–279 | `rel_err` | `np.abs((col_sums - expected)/(np.abs(expected)+1e-30))` | rel | max over sampled grid |
| `src/rates/compute_lmix.py` | 417–420 | `err` | `np.abs(actual_ratio - expected_ratio).max()` | abs (of ratio) | max over (n,l) |
| `src/rates/compute_lmix.py` | 472–476 | `rel_err` | `max_col_err / max_diag` | rel | pointwise |
| `src/rates/compute_K_CCC.py` | 239 | `max_err` | `np.max(np.abs(ratio_computed/ratio_expected - 1))*100` | rel % | max over TE_GRID |
| `src/rates/compute_term1_fraction.py` | 82 | `frac_err` | `abs(u0_F[IDX_3P_F]) / n_new[IDX_3P]` | rel | pointwise |
| `src/rates/solve_cr.py` | 593–594 | `maxerr` | `np.abs((sol.y[1:,-1]-n_ss[1:])/(n_ss[1:]+1e-60))`, filtered `>1e-3*max` | rel | max over filtered states |
| `src/parsers/qc2_ccc.py` | 111 | `E_error` | `abs(E_deexc_actual - E_deexc_target)` | abs | pointwise |
| `src/parsers/qc2_ccc.py` | 119 | `percent_error` | `100*abs(ratio - 1.0)` | rel % | pointwise |
| `src/validation/anderson_benchmark_qc.py` | 262–263 | `pct_err` | `(K_ccc/K_and - 1.0)*100.0` | rel signed % | pointwise; later `.abs()` at 478 |
| `src/validation/verify_plateau_slowmode.py` | 247–248 | `sup_err` | `np.abs(n_new_E_pred - n_new[E]).max()/np.abs(n_new[E]).max()` | rel | max over excited subset |
| `src/validation/verify_plateau_slowmode.py` | 304 | `lin_err` | `abs(tot/direct - 1)` | rel | pointwise |
| `src/analysis/unified_scaling.py` | 335–336 | `rel_err` | `(I_true - I_qss)/np.maximum(I_qss,1e-60)` | rel signed | time series |
| `src/analysis/unified_scaling.py` | 442–443 | `rel_err` | `abs_err / np.maximum(y_true,1e-60)` | rel | pointwise |
| `src/analysis/physics_tests.py` | 482 | `rel_err` | `(I_CR - I_QSS)/np.maximum(I_QSS,1e-60)*100` | rel signed % | time series |

`unified_scaling.py` defines `rel_err` twice with **different denominators**
(`I_qss` at 335, `y_true` at 443) — a second name collision independent of
`eps_step`. Not traced whether the two are ever compared.

---

## ITEM 7 — HARDCODED GRID ASSUMPTIONS

### What cr_context.py provides

- `CRContext.load(root=None, lgrid=None)` — auto-discovers repo root by walking up
  for `data/processed/cr_matrix/`, then loads `Te_grid_L.npy`, `ne_grid_L.npy`,
  `L_grid.npy`, and `state_index.csv` (tries
  `data/processed/collisions/K_exc_full/state_index.csv` first, then
  `data/processed/Radiative/state_index.csv`).
- Attributes: `ctx.te_grid`, `ctx.ne_grid`, `ctx.L_grid` (shape
  `(n_Te, n_ne, n_states, n_states)`), `ctx.labels`, `ctx.n_values`.
- Properties: `ctx.n_states` (= `L_grid.shape[-1]`, derived), `ctx.ground_index`
  (= `argmin(n_values)`, derived).
- `ctx.nearest_point(te_ev, ne_cm3)` -> `(ti, ni)` via argmin, no hardcoded indices.
- `ctx.validate()` cross-checks shapes and raises loudly on mismatch.

```
comm -23 <(find src -name "*.py" -type f | sort) <(grep -l "cr_context" $(find src -name "*.py" -type f) | sort)
```

**7 of 66 files import it:** `cr_context.py` itself plus
`verify_{divertor_map,plateau_gridmap,ramp_vs_step,timescales,boundary_descent,plateau_slowmode}.py`.
The other 59 do not.

### Non-importers that hardcode

| path | line | verbatim | hardcoded |
|---|---|---|---|
| `rates/diagnostic.py` | 4 | `ti, ni = 23, 5   # benchmark point` | ITER indices, no shape/label check |
| `rates/regenerate_figures.py` | 65 | `ti, ni = 23, 5   # ITER ref: Te~3eV, ne~1.39e14` | ITER indices (this script also loads L_grid directly) |
| `rates/Balmer_timescale_audit.py` | 251 | `benchmark = regime.iloc[((regime["Te_old_grid_eV"] - 2.947052).abs() + np.abs(np.log10(regime["ne_grid_cm-3"] / 1.389495e14))).argmin()]` | benchmark point values |
| `rates/check.py` | 7–8 | `ti = np.argmin(np.abs(np.logspace(0,1,50) - 3.0))` / `ni = np.argmin(np.abs(np.logspace(12,15,8) - 1e14))` | full grid formula re-derived; also `"Thesis M = 611"` at line 14 |
| `rates/qsscheck.py` | 6–7 | `ti = np.argmin(np.abs(np.logspace(0,1,50)-3.0))` / `ni = np.argmin(np.abs(np.logspace(12,15,8)-1e14))` | same |
| `rates/assemble_cr_matrix.py` | 80–81 | `TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)` / `NE_GRID = np.logspace(12, 15, 8)` | canonical root definition — not drift by itself |
| `rates/assemble_K_exc.py` | 76 | same `TE_GRID` line | duplicate |
| `rates/compute_K_CCC.py` | 85 | same | duplicate |
| `rates/compute_K_TICS.py` | 43 | same | duplicate |
| `rates/compute_K_VS.py` | 76 | same | duplicate |
| `rates/compute_lmix.py` | 102 | same | duplicate |
| `rates/ionization_rates.py` | 39 | same | duplicate |
| `rates/recombination_rates.py` | 95 | same | duplicate |
| `analysis/trapping_analysis.py` | 745–746 | `TE_GRID = …` / `NE_GRID = …` inside `except:` | fallback grid **defined after its first use** at line 741 |
| `validation/verify_bundling_psm20.py` | 163–164 | `Te = np.logspace(…, N_TE_EXPECTED)` / `ne = np.logspace(12,15,N_NE_EXPECTED)` | fallback grid if `Te_grid_L.npy` missing; prints a WARNING |
| `analysis/test_scaling.py` | 63–75 | `STATE_LABELS = [...]` then `N_STATES = 43` | full 43-state label list, not read from `state_index.csv` |
| `analysis/unified_scaling.py` | 50–52 | `N_STATES = 43` then `STATE_LABELS = [...]` | same list duplicated |
| `rates/solve_cr.py` | 136 | `self.n_states = 43` | set immediately after loading an L_grid of shape (50,8,43,43) |
| ~30 more | various | `np.zeros((43,43))`, shape asserts | systemic literal 43; reproduce with `grep -n "43, *43" src/**/*.py` |

### Mixed case — imports cr_context but ALSO hardcodes

| path | line | verbatim | note |
|---|---|---|---|
| `validation/verify_divertor_map.py` | 259 | `and abs(r["Te"] - 2.947052) < 1e-4 and abs(r["ne"] - 1.389495e14) / 1.389495e14 < 1e-3` | locates the ITER summary row by literal floats instead of `ctx.nearest_point()`, which cr_context provides for exactly this |

The other five cr_context importers hardcode nothing of this kind — their
remaining `linspace`/`logspace` calls are time arrays or index sampling built
from `ctx.te_grid`.

---

## ITEM 8 — ADAS / GATE D

### 8(a) gate_D verbatim

`grep -n '^def '` put `gate_D` at 275 and the next top-level `def` (`gate_E`) at
378 -> full span 275–377.

```python
275: def gate_D(arrs, n_ion=1e14):
276:     """
277:     Compare effective ionization (SCD) and recombination (ACD) from our
278:     CR model with ADAS SCD96/ACD96 effective coefficients.
279: 
280:     SCD_eff = sum_p K_ion(p)*n_p^ss / (ne * n_neutral)
281:     ACD_eff = sum_p [alpha_RR(p)*ne + alpha_3BR(p)*ne^2]*n_p^ss / (ne^2 * n_ion)
282: 
283:     Requires:
284:       data/processed/ADAS/SCD96_long.csv
285:       data/processed/ADAS/ACD96_long.csv
286:     """
287:     print("\n" + "="*60)
288:     print("GATE D — ADAS Effective Coefficient Comparison")
289:     print("="*60)
290: 
291:     if not os.path.exists(PATHS['SCD96']) or not os.path.exists(PATHS['ACD96']):
292:         print(f"  ADAS files not found:")
293:         print(f"    {PATHS['SCD96']}")
294:         print(f"    {PATHS['ACD96']}")
295:         print(f"  GATE D SKIPPED — upload ADAS files to run this gate.")
296:         return {'gate': 'D', 'passed': None, 'skipped': True}
297: 
298:     scd_adas = pd.read_csv(PATHS['SCD96'])
299:     acd_adas = pd.read_csv(PATHS['ACD96'])
300: 
301:     L = arrs['L_grid']; S = arrs['S_grid']
302:     Te = arrs['Te_grid']; ne = arrs['ne_grid']
303:     K_ion = arrs['K_ion']
304: 
305:     results = []
306: 
307:     for i_Te in range(len(Te)):
308:         for i_ne in range(len(ne)):
309:             # FIX 1: use quasi-neutral n_ion = ne at each grid point.
310:             # Using fixed n_ion=1e14 with ne=1e12 puts the model in a deep
311:             # recombining phase (n_ion/ne=100), inflating Rydberg populations
312:             # and making SCD_model ~100x too large. ADAS SCD assumes quasi-
313:             # neutrality (n_ion ~= ne), so this fix makes the comparison valid.
314:             n_ion_local = ne[i_ne]
315: 
316:             Lmat = L[i_Te, i_ne]
317:             Svec = S[i_Te, i_ne] * n_ion_local
318:             n_ss = steady_state(Lmat, Svec)
319:             n_neutral = n_ss.sum()
320:             if n_neutral <= 0: continue
321: 
322:             # SCD_eff from our model
323:             # SCD = sum_p K_ion(p)*n_ss(p) / n_neutral  [cm^3/s]
324:             # ne cancels: K_ion*ne*n_ss / (ne*n_neutral) = K_ion*n_ss/n_neutral
325:             ioniz_rate = np.sum(K_ion[:, i_Te] * n_ss)
326:             SCD_model  = ioniz_rate / n_neutral
327: 
328:             # FIX 2: 2D interpolation — select ADAS rows near ne[i_ne],
329:             # then interpolate in Te. Avoids averaging over wrong ne values.
330:             # The original code averaged SCD over ALL ne (5e7 to 2e15),
331:             # diluting the mean toward the low-ne coronal limit.
332:             try:
333:                 mask = (scd_adas['ne_cm3'] / ne[i_ne]).between(0.5, 2.0)
334:                 sub  = scd_adas[mask].sort_values('Te_eV')
335:                 if len(sub) >= 2:
336:                     scd_at = float(np.interp(Te[i_Te],
337:                                              sub['Te_eV'].values,
338:                                              sub['SCD_cm3_s'].values))
339:                 else:
340:                     scd_at = np.nan
341:             except Exception:
342:                 scd_at = np.nan
343: 
344:             if np.isfinite(scd_at) and scd_at > 0:
345:                 eta = SCD_model / scd_at
346:                 results.append({
347:                     'Te_eV':     Te[i_Te],
348:                     'ne_cm3':    ne[i_ne],
349:                     'SCD_model': SCD_model,
350:                     'SCD_ADAS':  scd_at,
351:                     'eta_SCD':   eta,
352:                     'passed':    0.5 <= eta <= 2.0,
353:                 })
354: 
355:     if not results:
356:         print("  Could not interpolate ADAS values — check CSV column names.")
357:         return {'gate': 'D', 'passed': None, 'skipped': True}
358: 
359:     df = pd.DataFrame(results)
360:     n_pass = df['passed'].sum()
361:     frac   = n_pass / len(df)
362:     passed = frac >= 0.7
363: 
364:     print(f"  Points compared: {len(df)}")
365:     print(f"  Factor-2 agreement: {n_pass}/{len(df)} ({frac*100:.0f}%)")
366:     print(f"  Status: {'PASS' if passed else 'FAIL (check Te range)'}")
367:     print()
368:     print(f"  {'Te':6s}  {'ne':10s}  {'SCD_model':12s}  {'SCD_ADAS':12s}  {'eta':8s}  pass")
369:     for _, row in df[::max(1,len(df)//10)].iterrows():
370:         print(f"  {row['Te_eV']:6.2f}  {row['ne_cm3']:10.2e}  "
371:               f"{row['SCD_model']:12.4e}  {row['SCD_ADAS']:12.4e}  "
372:               f"{row['eta_SCD']:8.3f}  {'Y' if row['passed'] else 'N'}")
373: 
374:     return {'gate': 'D', 'passed': passed, 'pass_frac': frac, 'df': df}
```

Referenced definitions elsewhere in the file:

```python
44: IH_RYDBERG = 13.605693   # eV
45: L_CHAR     = 'SPDFGHIJKL'
```

```python
59: PATHS = {
60:     'L_grid':    'data/processed/cr_matrix/L_grid.npy',
61:     'S_grid':    'data/processed/cr_matrix/S_grid.npy',
62:     'Te_grid':   'data/processed/cr_matrix/Te_grid_L.npy',
63:     'ne_grid':   'data/processed/cr_matrix/ne_grid_L.npy',
64:     'K_ion':     'data/processed/collisions/tics/K_ion_final.npy',
65:     'K_exc':     'data/processed/collisions/K_exc_full/K_exc_full.npy',
66:     'K_deexc':   'data/processed/collisions/K_exc_full/K_deexc_full.npy',
67:     'K_exc_meta':'data/processed/collisions/K_exc_full/K_exc_meta.csv',
68:     'SCD96':     'data/processed/adas/SCD96_interpolated.csv',
69:     'ACD96':     'data/processed/adas/ACD96_interpolated.csv',
70: }
71: OUT_DIR = 'validation'
```

```python
84: def steady_state(L_mat, S_vec):
85:     """Compute n_ss = -L^{-1} * S, floor at 0."""
86:     return np.maximum(np.linalg.solve(L_mat, -S_vec), 0.0)
```

`arrs` is built by `load_arrays()` (lines 76–81), loading `L_grid`, `S_grid`,
`Te_grid`, `ne_grid`, `K_ion`, `K_exc`, `K_deexc`, `K_exc_meta` from `PATHS`.

**Three structural notes, reported not corrected:**

1. **The docstrings name files the code does not read.** Lines 283–285 say
   `data/processed/ADAS/SCD96_long.csv`; the module docstring (24–29) says
   `data/processed/adas/scd96_h_long.csv`. The code reads `PATHS['SCD96']` =
   `data/processed/adas/SCD96_interpolated.csv`. Those are different files with
   different column names — see 8(b).
2. `acd_adas` is loaded at line 299 and **never used**. The ACD half of the gate's
   stated purpose (docstring line 281) is not implemented; only SCD is compared.
3. The `n_ion=1e14` signature default is shadowed at line 314 by
   `n_ion_local = ne[i_ne]` and never used.

### 8(b) data/processed/adas/ contents

| path | size | rows (`wc -l`) | mtime |
|---|---|---|---|
| `.DS_Store` | 6148 B | — | 2026-03-22 19:03:18 |
| `acd96_h_long.csv` | 46665 B | 697 (696 + header) | 2026-02-21 03:35:06 |
| `ACD96_interpolated.csv` | 45782 B | 697 | 2026-03-22 18:34:02 |
| `report_adas.md` | 977 B | text | 2026-03-19 02:30:05 |
| `scd96_h_long.csv` | 46251 B | 697 | 2026-02-21 03:35:06 |
| `SCD96_interpolated.csv` | 45420 B | 697 | 2026-03-22 18:34:02 |

Headers verbatim (`head -3`):

```
acd96_h_long.csv:        Te_eV,ne_cm3,log10_K,K_cm3_s
ACD96_interpolated.csv:  Te_eV,ne_cm3,log10_K,ACD_cm3_s
scd96_h_long.csv:        Te_eV,ne_cm3,log10_K,K_cm3_s
SCD96_interpolated.csv:  Te_eV,ne_cm3,log10_K,SCD_cm3_s
```

The `*_h_long.csv` files use `K_cm3_s`; only the `*_interpolated.csv` files carry
`SCD_cm3_s`/`ACD_cm3_s`, which is what line 338 expects. The code is consistent
with the file it loads; the docstrings are not.

`report_adas.md` (pre-existing in the repo) records that eta = SCD_model/SCD_ADAS
lands in 1e2–1e4 scaling as ne^1.5, attributing it to gross-vs-net GCR coefficient
mismatch, and concludes "the two quantities measure different physics and cannot
be directly compared."

### 8(c) Who references adas

`grep -rniI 'adas' src/`:

| script | lines | role |
|---|---|---|
| `src/parser_adasf11.py` | 24, 102, 107, 186–189, 197, 221–226 | **writes** `adas/{scd96,acd96}_h_long.csv` from raw `.dat` |
| `src/rates/prepare_adas.py` | 4, 6, 11, 13, 17 | **writes** `adas/{SCD96,ACD96}_interpolated.csv` |
| `src/validation/validate_gates.py` | 24–29, 39, 70–71, 274–372, 476 | reads `*_interpolated.csv` |
| `src/adas_interpolator.py` | 3, 5, 7, 54–55, 64 | reads raw `data/raw/adas/*.dat` |
| `src/week2_timescale_map.py` | 3, 28, 50–55 | reads raw `.dat` |
| `src/config/paths.py` | 64, 70, 97–99, 130, 152, 203–204 | **a second, parallel path registry** — unused by gate_D |
| `src/analysis/escape_factor.py`, `src/analysis/trapping_analysis.py` | 6, 17, 41, 142, 194 / 6 | ADAS214 citations only, no I/O |

Chain: `parser_adasf11.py` -> `prepare_adas.py` -> `gate_D`. The mtimes
(02-21 -> 03-22) run in that order.

---

## ITEM 9 — FIGURE PROVENANCE

141 files (excluding `.DS_Store`). `.pdf`/`.png` pairs sharing a producer are
merged into one row; mtimes differ by <=1 s where noted.

| figure | mtime | producer | newest input |
|---|---|---|---|
| `fig1_epsilon_traces.png` | 03-28 21:48:16 | `analysis/plot_results.py:181` | **08-23 11:17:01** |
| `fig2_breakdown_map.png` | 03-28 21:48:17 | `plot_results.py:241` | **08-23 11:17:01** |
| `fig3_timescales.png` | 03-28 21:48:17 | `plot_results.py:296` | **08-23 11:17:01** |
| `fig4_eps_step.png` | 03-28 21:48:17 | `plot_results.py:324` | **08-23 11:17:01** |
| `fig5_populations.png` | 03-28 21:48:17 | `plot_results.py:379` | **08-23 11:17:01** |
| `fig6_regime_map.png` | 03-28 21:48:18 | `plot_results.py:447` | **08-23 11:17:01** |
| `fig7_eps_scaling.png` | 03-28 21:48:18 | `plot_results.py:529` | **08-23 11:17:01** |
| `S_criterion_map.{pdf,png}` | 05-20 23:04:27 | `rates/S_criterion.py:202` | 07-21 20:44:53 |
| `S_criterion_collapse.{pdf,png}` | 05-20 23:04:27 | `S_criterion.py:243` | 07-21 20:44:53 |
| `S_criterion_fixed_map.{pdf,png}` | 05-10 02:46:08 | `S_criterion_fixed.py:509` | 07-21 20:44:53 |
| `S_criterion_fixed_collapse_linear.{pdf,png}` | 05-10 02:46:08 | `S_criterion_fixed.py:594` (stem arg line 751) | 07-21 20:44:53 |
| `S_criterion_fixed_collapse_exp.{pdf,png}` | 05-10 02:46:08 | `S_criterion_fixed.py:594` (stem arg line 756) | 07-21 20:44:53 |
| `S_3P_map.{pdf,png}` | 05-20 23:28:16 | `S_criterion_3P.py:297` | 07-21 20:44:53 |
| `S_3P_collapse.{pdf,png}` | 05-20 23:28:16/17 | `S_criterion_3P.py:330` | 07-21 20:44:53 |
| `mz_fig6_decomposition.{pdf,png}` | 04-12 01:13:37 | `compute_mz_decomposition.py:246` | 07-21 20:44:53 |
| `mz_fig1_kernel_ITER.{pdf,png}` | 04-12 03:28:19 | `regenerate_figures.py:115` | 07-21 20:44:53 |
| `mz_fig3_M_comparison.{pdf,png}` | 04-11 21:48:34 | `mori_zwanzig_weekc.py:301` — **not** `regenerate_figures.py:152` | **08-23 11:17:01** |
| `mz_fig2_tauK_map.{pdf,png}` | 04-11 21:48:34 | `mori_zwanzig_weekc.py:247` | **08-23 11:17:01** |
| `mz_fig4_validation.{pdf,png}` | 04-11 21:48:34 | `mori_zwanzig_weekc.py:335` | **08-23 11:17:01** |
| `mz_fig5_scaling.{pdf,png}` | 04-11 21:48:35 | `mori_zwanzig_weekc.py:391` | **08-23 11:17:01** |
| `mz_fig7_term1_fraction.{pdf,png}` | 04-12 02:06:44 | `compute_term1_fraction.py:235` | **08-23 11:17:01** |
| `Balmer_transient_ratio_DTe_{p0p30,p0p60,p1p00,p2p00,p3p00}.{pdf,png}` (10) | 05-10 04:48:34–36 | `Balmer_transient_ratio.py:615` | 07-21 20:44:53 |
| `Balmer_transient_absolute_errors_DTe_{same 5}.{pdf,png}` (10) | 05-10 04:48:34–36 | `Balmer_transient_ratio.py:639` | 07-21 20:44:53 |
| `Balmer_transient_ratio_errors_overlay.{pdf,png}` | 05-10 04:48:36 | `Balmer_transient_ratio.py:665` | 07-21 20:44:53 |
| `Halpha_sensitivity_map.{pdf,png}` | 05-10 02:53:12 | `Halpha_sensitivity.py:498` | 07-21 20:44:53 |
| `Halpha_sensitivity_collapse_{linear,exp}.{pdf,png}` (4) | 05-10 02:53:12 | `Halpha_sensitivity.py:579` (stems at 711, 713) | 07-21 20:44:53 |
| `Balmer_v2_{Halpha,Hbeta,Halpha_over_Hbeta}_sensitivity_map.{pdf,png}` (6) | 05-10 03:26:42–44 | `Balmer_ratio_sensitivity.py:484` | 07-21 20:44:53 |
| `Balmer_v2_{...}_collapse_{linear,exp}.{pdf,png}` (12) | 05-10 03:26:42–44 | `Balmer_ratio_sensitivity.py:548` | 07-21 20:44:53 |
| `Balmer_timescale_last_gt10_vs_tau_slow.{pdf,png}` | 05-10 05:37:00 | `Balmer_timescale_audit.py:318/319` | 05-10 05:20:17 direct; 07-21 20:44:53 whole-script |
| `Balmer_timescale_peak_error_vs_tau_slow.{pdf,png}` | 05-10 05:37:00 | `Balmer_timescale_audit.py:340/341` | as above |
| `Balmer_timescale_peak_error_vs_M.{pdf,png}` | 05-10 05:37:00 | `Balmer_timescale_audit.py:363/364` | as above |
| `Balmer_regime_peak_ratio_error_DTe_p0p60.{pdf,png}` | 05-10 05:20:18 | `Balmer_transient_regime_heatmap.py:260` | 07-21 20:44:53 |
| `Balmer_regime_tau_slow_DTe_p0p60.{pdf,png}` | 05-10 05:20:18 | `…heatmap.py:278` | 07-21 20:44:53 |
| `Balmer_regime_M_DTe_p0p60.{pdf,png}` | 05-10 05:20:19 | `…heatmap.py:287` | 07-21 20:44:53 |
| `Balmer_regime_last_gt10_DTe_p0p60_fixed.{pdf,png}` | 05-10 05:29:18 | `plot_regime_persistence_fixed.py:216` (no `--overwrite`) | 05-10 05:20:17 |
| `Balmer_regime_last_gt10_DTe_p0p60.{pdf,png}` | 05-10 05:29:50 | **inferred** same script with `--overwrite` — mtime ordering only, not confirmed | 05-10 05:20:17 |
| `Balmer_robustness_peak_ratio_error_vs_Te.{pdf,png}` | 05-10 04:59:43/44 | `Balmer_transient_robustness_sweep.py:209` | 07-21 20:44:53 |
| `Balmer_robustness_duration_vs_Te.{pdf,png}` | 05-10 04:59:44 | `…sweep.py:228` | 07-21 20:44:53 |
| `Balmer_robustness_density_sweep.{pdf,png}` | 05-10 04:59:44 | `…sweep.py:267` | 07-21 20:44:53 |
| `Balmer_robustness_peak_ratio_scatter.{pdf,png}` | 05-10 04:59:44 | `…sweep.py:291` | 07-21 20:44:53 |
| `fig_boundary_descent.{pdf,png}` | **08-06 02:46:33** | `verify_boundary_descent.py:215/216` | 07-21 20:44:53 |
| `anderson_benchmark_full.png` | 03-22 17:43:10 | `anderson_benchmark_qc.py:493` | 07-02 01:40:11 |
| `week2/K_CCC_diagnostic.png` | 07-02 01:40:11 | `compute_K_CCC.py:310` | 03-22 01:15:36 |
| `week2/ccc_qc_report.png` | 06-14 14:54:30 | `parsers/qc_ccc.py:375` — **input CSV is `sys.argv[1]`, cannot be determined statically** | not determinable |
| `week2/ccc_qc_detailed_balance.png` | 03-13 00:15:35 | `parsers/qc2_ccc.py:310` — input CSV is `sys.argv[1]`; `output_dir` hardcoded | not determinable |
| `Balmer_{Halpha,Hbeta,Halpha_over_Hbeta}_{sensitivity_map,collapse_linear,collapse_exp}.{pdf,png}` (18) | 05-10 03:12:06–09 | **no producer found** | n/a |
| `ADAS_validation.png` | 02-15 05:29:07 | **no producer found** | n/a |
| `week1/plot{1,2,3a,3b,4}_*.png` (5) | 03-14 01:03:42–43 | **no producer found** | n/a |
| `paper/paper_fig_balmer_*_DTe_0p609.{pdf,png}` + `…overlay_all_DTe.{pdf,png}` (6) | 05-10 04:49:32 | **no producer found** | n/a |

### Notes on the undeterminable and anomalous cases

- **`mz_fig3_M_comparison`** — two scripts define a savefig for this exact name.
  `regenerate_figures.py`'s docstring says it exists to remove the stale 25 ns
  tau_relax line and the M_thesis-vs-M_MZ panel; its `__main__` (158–174) calls
  both `fig1_kernel()` and `fig3_M_MZ_single()`. `mz_fig1_kernel_ITER` carries a
  04-12 03:28:19 mtime matching that run; `mz_fig3_M_comparison` still carries
  04-11 21:48:34 from `mori_zwanzig_weekc.py`. **The file on disk is the
  uncorrected version**, despite the corrective script having apparently run. Why
  the second write did not land cannot be determined without running it.
- **`mori_zwanzig_weekc.py:270` writes `mz_fig3_M_MZ`, which does not exist
  anywhere in `figures/`.**
- **The 18 non-"v2" `Balmer_*` figures**: `grep -rn` for the literal stems returns
  nothing in `src/`. They are dated 03:12:06–09, 14 minutes *before*
  `Balmer_ratio_sensitivity.py`'s own save (03:26:41), with the "v2"-prefixed set
  written at 03:26:42–44. Consistent with the script having been edited in place
  to add the prefix; the code that wrote the old names no longer exists.
- **`paper/…_DTe_0p609`**: no producer; `grep` for `paper_fig`, `figures/paper`,
  `0p609` returns nothing in `src/`. Dated 56 s after the
  `Balmer_transient_ratio.py` run. A resemblance to that script's `delta_actual`
  vs `delta_nominal` distinction is **speculation only** — unconfirmable.
- `fig_boundary_descent` is the **only** figure in the repo that postdates every
  input it loads.
- 16 figure files are stale against `validation/{M_grid,tau_QSS_grid,tau_relax_grid}.npy`,
  which carry mtime **2026-08-23 11:17:01**.
- The `/mnt/user-data/uploads/` fallback in `anderson_benchmark_qc.py` was probed
  with `ls` and does not exist; the primary local CSV does, so the fallback is inert.

---

## ITEM 10 — ORPHANS

Method: `find data/processed validation -type f | grep -v .DS_Store`, then
`grep -rl -- "$basename" src/` per file; zero hits => candidate.

**Stated limitation:** this catches only files whose basename appears nowhere at
all. Files written by a live script and simply never read back are listed
separately below.

### (a) Zero-hit orphans

| file | mtime | writer | status |
|---|---|---|---|
| `validation/timescales_unfiltered_CHECK.npz` | 08-22 14:05:06 | no script — manual run, per commit `3a5eb8c`: *"unconditional recomputation of tau_QSS/tau_relax/M, written for comparison only. Confirms 19 of 400 points corrupted by the eigs < -1.0 filter. Nothing overwritten. Filter fix NOT yet applied."* | no reader — **git-committed quantification of the truncation bug at 19/400 = 4.75% of the grid** |
| `validation/plateau_window_check.{csv,txt}` | 08-22 23:32:20 | `verify_plateau_window.py` — **absent from the tree and from `git log --all`**; named only in commit `d64f0fb`'s message | no reader; writer confirmed gone |
| `validation/{tau_QSS_grid,tau_relax_grid,M_grid}.npy_FILTERED_20260721`, `breakdown_map.csv_…`, `qss_analysis_summary.txt_…`, `epsilon_traces.npz_…` (6) | 07-21 21:02:14 — 17 min after L_grid rebuild | manual rename of a `qss_analysis.py` run | no reader; a preserved snapshot of the filtered outputs |
| `validation/gate_{B,C,D}.csv` | 08-23 11:17:01 | `validate_gates.py:460` | no reader in src/ |
| `mori_zwanzig/mz_decomposition_{t,full,termI,termII}.npy` | 04-12 01:06:58 | none — 7 min older than the `mz_decomp_*` files; earlier naming of the same script | orphaned |
| `sensitivity/Balmer_ratio_sensitivity_results.npz` (no `_v2`) | 05-10 03:12:06 | none — 15 min older than the `_v2` file | orphaned pre-v2 leftover (matches the 03:12 non-"v2" figures in item 9) |
| `collisions/ccc/K_CCC_table.npy` | 03-13 21:16:51 | none — the current script writes the split `exc`/`deexc` tables | orphaned, oldest artifact found |
| `sensitivity/regime/…DTe_p0p60.tex` | 05-10 05:20:17 | `Balmer_transient_regime_heatmap.py:333` (dynamic f-string) | no reader — but the companion `.csv` **is** read by `Balmer_timescale_audit.py:69,221`; only the `.tex` is orphaned |
| `sensitivity/paper/paper_balmer_transient_summary_table.{csv,tex}` | 05-10 04:49:07 | **none anywhere in src/** | no reader; provenance untraceable from the current tree (same 04:49 batch as the unproducible `figures/paper/` files) |
| ~40 `.png` under `validation/{unified_scaling_validation,scaling_tests,physics_tests,unified_scaling_v2}/` | Apr–Aug | `unified_scaling.py`, `test_scaling.py`, `physics_tests.py` | terminal figure outputs — no reader expected |
| `*/report.md`, `Radiative/validation.txt`, `collisions/*.md`, `adas/report_adas.md`, `ccc/README_K_CCC.md`, `CCC_QC_Report.docx`, `Lotz_1968.pdf` | various | reports / manual documents | terminal documentation, not machine-read |

**Single-producer, no external reader** (not caught by the basename scan,
traceable via item 4): `sensitivity/S_grid.npy`, `p_max_grid.npy`,
`Halpha_sensitivity_results.npz`, `S_criterion_fixed_results.npz`,
`eps_3P_results.npz`, `Balmer_ratio_sensitivity_v2_*`,
`Balmer_transient_ratio_summary.csv`, `mz_decomp_*.npy`.

### (b) Scripts reading files that do not exist

| script | line | path | status |
|---|---|---|---|
| `rates/mori_zwanzing_weekb.py` | 239 | `validation/tau_QSS_grid.npy` (guarded by `.exists()`) | **exists today** — flagged only because lines 244–245 silently substitute `tau_relax_MZ` "as a proxy (underestimate)" if absent |
| `rates/plot_regime_persistence_fixed.py` | 139 | `pd.read_csv(args.csv)` | dynamic CLI path — cannot be determined statically |
| all other literal reads traced in the item-4 closure | — | — | all targets present |

Not exhaustive across all 66 files: this covers every literal path traced while
building the L_grid closure, plus the one defensive-check case. `check_mz.py`,
`qc_ccc.py`, `qc2_ccc.py`, `pre_assembly_check.py` were not line-by-line audited
for broken reads.

---

## DOCUMENT-VS-DISK CONFLICTS

Surfaced by the survey; recorded because CLAUDE.md's ground-truth hierarchy puts
code and disk above documents. **No file was changed.**

| # | Document claim | Disk says |
|---|---|---|
| 1 | CLAUDE.md known-issues table: `src/validation/qss_analysis.py` contains `eigs = eigs[eigs < -1.0]` | `grep -n "\-1\.0" src/validation/qss_analysis.py` returns nothing. That file's filter (line 137) is `neg = eigs[eigs < 0.0]`. The literal `eigs[eigs < -1.0]` is in `src/rates/solve_cr.py:269` and `src/rates/check_mz.py:10`. The issue is real; the location in the table is not. |
| 2 | CLAUDE.md: "L_grid.npy regenerated 14 Jul 2026 after the l-mixing F(U_m) correction" | `stat` gives `2026-07-21 20:44:53` — seven days later. Only one copy of the file exists. |
| 3 | `validate_gates.py` gate_D docstrings name `data/processed/ADAS/SCD96_long.csv` and `data/processed/adas/scd96_h_long.csv` | The code reads `data/processed/adas/SCD96_interpolated.csv`, a different file with a different column name (`SCD_cm3_s` vs `K_cm3_s`). Code is self-consistent; docstrings are stale. |
| 4 | gate_D docstring line 281 defines an ACD_eff comparison | `acd_adas` is loaded at line 299 and never used. Only SCD is compared. |

---

*Generated 2026-08-23 by four read-only verifier agents. Nothing was executed;
no project script was run or imported; no repository file was modified.*
