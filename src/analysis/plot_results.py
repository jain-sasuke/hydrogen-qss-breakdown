"""
plot_results.py
===============
Publication-quality figures for the hydrogen CR QSS breakdown analysis.
All 7 figures in one file. Run from repo root:

    python src/analysis/plot_results.py

Outputs saved to figures/:
  fig1_epsilon_traces.png  — Two-stage QSS relaxation epsilon(t)
  fig2_breakdown_map.png   — QSS breakdown map at 4 ITER timescales
  fig3_timescales.png      — tau_QSS and M maps
  fig4_eps_step.png        — eps_step sensitivity map
  fig5_populations.png     — Population distribution vs Saha-Boltzmann
  fig6_regime_map.png      — Three-regime classification
  fig7_eps_scaling.png     — eps vs M scatter + Te scaling law

Style: Physical Review E — compact, serif, 300 dpi.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR  = 'figures'
DATA_DIR = 'validation'
CR_DIR   = 'data/processed/cr_matrix'   # for L_grid, S_grid

# ── Global grids ───────────────────────────────────────────────────────────────
Te_grid = np.logspace(np.log10(1.0), np.log10(10.0), 50)
ne_grid = np.logspace(12, 15, 8)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          10,
    'axes.labelsize':     11,
    'axes.titlesize':     11,
    'legend.fontsize':    9,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.5,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.top':          True,
    'ytick.right':        True,
})

C = {
    'Te1':   '#1f77b4',   # blue   — Te=1 eV
    'Te3':   '#d62728',   # red    — Te=3 eV (ITER)
    'Te10':  '#2ca02c',   # green  — Te=10 eV
    'break': '#d62728',   # red contour
    'ITER':  '#ff7f0e',   # orange ITER box
}


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(data_dir=DATA_DIR):
    return {
        'M':   np.load(f'{data_dir}/M_grid.npy'),
        'tQ':  np.load(f'{data_dir}/tau_QSS_grid.npy'),
        'tr':  np.load(f'{data_dir}/tau_relax_grid.npy'),
        'bd':  pd.read_csv(f'{data_dir}/breakdown_map.csv'),
        'eps': np.load(f'{data_dir}/epsilon_traces.npz'),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def add_ITER_box(ax):
    """Orange dashed box: ITER divertor regime Te=1-5 eV, ne=1e13-1e15."""
    rect = Rectangle((13, 1), 2, 4,
                      fill=False, edgecolor=C['ITER'],
                      linewidth=1.8, linestyle='--', zorder=5)
    ax.add_patch(rect)
    ax.text(14.0, 5.3, 'ITER\ndivertor', color=C['ITER'],
            fontsize=8, ha='center', va='bottom', fontweight='bold')


def build_Z(bd, col):
    """Pivot breakdown_map column into (Te, ne) 2-D array."""
    Te_vals = np.sort(bd['Te_eV'].unique())
    ne_vals = np.sort(bd['ne_cm3'].unique())
    Z = np.zeros((len(Te_vals), len(ne_vals)))
    for i, Tv in enumerate(Te_vals):
        for j, nv in enumerate(ne_vals):
            r = bd[(abs(bd['Te_eV']-Tv) < 0.01) & (abs(bd['ne_cm3']-nv) < nv*0.01)]
            if len(r):
                Z[i, j] = r[col].values[0]
    return Z, Te_vals, np.log10(ne_vals)


def get_ss(L, S, i_Te, i_ne, n_ion=1e14):
    Lm = L[i_Te, i_ne]; Sv = S[i_Te, i_ne] * n_ion
    return np.maximum(np.linalg.solve(Lm, -Sv), 0)


# ── Figure 1 — epsilon(t) traces ──────────────────────────────────────────────
def fig1_epsilon_traces(data, out_dir):
    """Three-panel two-stage relaxation after +0.6 eV Te step."""
    eps = data['eps']; tQ_g = data['tQ']; tr_g = data['tr']

    ti1  = np.argmin(np.abs(Te_grid - 1.0));   ni12 = 0
    ti3  = np.argmin(np.abs(Te_grid - 3.0));   ni14 = np.argmin(np.abs(ne_grid - 1e14))
    ti10 = np.argmin(np.abs(Te_grid - 10.0));  ni15 = 7

    panels = [
        ('Te1eV_ne1e12',
         C['Te1'],
         r'$T_e=1\,\mathrm{eV},\;n_e=10^{12}\,\mathrm{cm}^{-3}$',
         tQ_g[ti1, ni12], tr_g[ti1, ni12]),
        ('Te3eV_ne1e14_(ITER)',
         C['Te3'],
         r'$T_e=3\,\mathrm{eV},\;n_e=10^{14}\,\mathrm{cm}^{-3}$ (ITER)',
         tQ_g[ti3, ni14], tr_g[ti3, ni14]),
        ('Te10eV_ne1e15',
         C['Te10'],
         r'$T_e=10\,\mathrm{eV},\;n_e=10^{15}\,\mathrm{cm}^{-3}$',
         tQ_g[ti10, ni15], tr_g[ti10, ni15]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)

    for ax, (key, color, title, tQ_val, tr_val) in zip(axes, panels):
        t = eps[f't_{key}']; e = eps[f'eps_{key}']
        mask = t > 0; t = t[mask]; e = e[mask]

        ax.semilogx(t, e, color=color, lw=2.0, zorder=3)
        ax.fill_between(t, 0, e, alpha=0.12, color=color)

        if tr_val > 0:
            ax.axvline(tr_val, color='gray',  lw=1.0, ls='--', zorder=2,
                       label=r'$\tau_\mathrm{relax}$')
        if tQ_val > 0:
            ax.axvline(tQ_val, color='black', lw=1.0, ls=':',  zorder=2,
                       label=r'$\tau_\mathrm{QSS}$')

        ax.axhline(0.5, color=C['break'],  lw=0.8, ls='-.', alpha=0.7,
                   label=r'$\epsilon=0.5$')
        ax.axhline(0.1, color='steelblue', lw=0.8, ls='-.', alpha=0.5)

        ax.set_xlabel(r'Time $t$ [s]')
        ax.set_ylabel(r'QSS error $\epsilon_\mathrm{max}$')
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_ylim(-0.05, 1.10)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

        if e.max() > 0.05:
            if tr_val > 0:
                ax.text(tr_val * 0.3, 0.03, 'Stage 1', fontsize=7.5,
                        color='gray', ha='center', va='bottom', style='italic')
            if tQ_val > 0 and tr_val > 0 and tQ_val > tr_val * 2:
                ax.text(np.sqrt(tr_val * tQ_val) * 5, 0.03, 'Stage 2',
                        fontsize=7.5, color='gray', ha='center',
                        va='bottom', style='italic')

        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8, handlelength=1.5)

    fig.suptitle(
        r'Two-stage QSS relaxation after $\Delta T_e = +0.6\,\mathrm{eV}$ step',
        y=1.02, fontsize=11)
    plt.tight_layout(w_pad=1.0)
    path = f'{out_dir}/fig1_epsilon_traces.png'
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 2 — breakdown map 2×2 ──────────────────────────────────────────────
def fig2_breakdown_map(data, out_dir):
    """2×2 eps_eff heatmaps at four ITER transient timescales."""
    bd = data['bd']
    timescales = [
        ('eps_ELM_crash',       r'ELM crash ($\tau=100\,\mu\mathrm{s}$)'),
        ('eps_fast_detachment', r'Fast detachment ($\tau=1\,\mathrm{ms}$)'),
        ('eps_slow_detachment', r'Slow detachment ($\tau=10\,\mathrm{ms}$)'),
        ('eps_ELM_interELM',    r'Inter-ELM ($\tau=100\,\mathrm{ms}$)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
    cmap = plt.cm.RdYlBu_r

    for ax, (col, title) in zip(axes.flatten(), timescales):
        Z, Te_vals, log_ne = build_Z(bd, col)
        Z = np.clip(Z, 0, 1.5)

        ax.pcolormesh(log_ne, Te_vals, Z, cmap=cmap, vmin=0, vmax=1.5, shading='auto')
        try:
            cs = ax.contour(log_ne, Te_vals, Z, levels=[0.5],
                            colors=[C['break']], linewidths=1.8, zorder=4)
            ax.clabel(cs, fmt=r'$\epsilon=0.5$', fontsize=7.5,
                      inline=True, use_clabeltext=True)
        except Exception:
            pass
        try:
            ax.contour(log_ne, Te_vals, Z, levels=[0.1],
                       colors=['white'], linewidths=0.8, linestyles='--', zorder=4)
        except Exception:
            pass

        add_ITER_box(ax)
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.set_xlim(12, 15); ax.set_ylim(1, 10)

    for ax in axes[1]:
        ax.set_xlabel(r'$\log_{10}(n_e\,[\mathrm{cm}^{-3}])$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$T_e$ [eV]')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1.5))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(r'QSS error $\epsilon_\mathrm{eff}$', fontsize=10)
    cb.ax.axhline(0.5, color=C['break'], linewidth=1.5)

    fig.suptitle('QSS breakdown map: effective error at ITER transient timescales',
                 y=0.98, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.94, bottom=0.10)
    path = f'{out_dir}/fig2_breakdown_map.png'
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 3 — timescale maps ──────────────────────────────────────────────────
def fig3_timescales(data, out_dir):
    """tau_QSS map (left) and M = tau_QSS/tau_relax map (right)."""
    tQ = data['tQ']; M = data['M']
    log_ne = np.log10(ne_grid)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Left: tau_QSS in ms (log scale)
    ax = axes[0]
    Z1 = np.log10(tQ * 1e3)
    im1 = ax.pcolormesh(log_ne, Te_grid, Z1, cmap='plasma_r',
                        vmin=Z1.min(), vmax=Z1.max(), shading='auto')
    cb1 = fig.colorbar(im1, ax=ax, pad=0.02)
    cb1.set_label(r'$\log_{10}(\tau_\mathrm{QSS}$ [ms])', fontsize=10)
    try:
        cs1 = ax.contour(log_ne, Te_grid, Z1, levels=[-2,-1,0,1,2],
                         colors='white', linewidths=0.8, alpha=0.7)
        ax.clabel(cs1,
                  fmt={-2:r'$1\,\mu$s',-1:r'$10\,\mu$s',0:r'$100\,\mu$s',
                       1:r'$1\,$ms',2:r'$10\,$ms'},
                  fontsize=7, inline=True)
    except Exception:
        pass
    add_ITER_box(ax)
    ax.set_xlabel(r'$\log_{10}(n_e\,[\mathrm{cm}^{-3}])$')
    ax.set_ylabel(r'$T_e$ [eV]')
    ax.set_title(r'Ionisation balance timescale $\tau_\mathrm{QSS}$', fontsize=10)

    # Right: M map (log scale)
    ax2 = axes[1]
    Z2 = np.log10(M)
    im2 = ax2.pcolormesh(log_ne, Te_grid, Z2, cmap='viridis',
                         vmin=1.5, vmax=5.5, shading='auto')
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cb2.set_label(r'$\log_{10}(M)$', fontsize=10)
    try:
        cs2 = ax2.contour(log_ne, Te_grid, Z2, levels=[2,3,4,5],
                          colors='white', linewidths=0.8, alpha=0.7)
        ax2.clabel(cs2, fmt={2:'M=100',3:'M=1000',4:r'$10^4$',5:r'$10^5$'},
                   fontsize=7.5, inline=True)
    except Exception:
        pass
    add_ITER_box(ax2)
    ax2.set_xlabel(r'$\log_{10}(n_e\,[\mathrm{cm}^{-3}])$')
    ax2.set_ylabel(r'$T_e$ [eV]')
    ax2.set_title(r'Memory metric $M=\tau_\mathrm{QSS}/\tau_\mathrm{relax}$', fontsize=10)

    plt.tight_layout()
    path = f'{out_dir}/fig3_timescales.png'
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 4 — eps_step sensitivity map ───────────────────────────────────────
def fig4_eps_step(data, out_dir):
    """eps_step(Te,ne): initial QSS error after +0.6 eV Te step."""
    bd = data['bd']
    Z, Te_vals, log_ne = build_Z(bd, 'eps_step')

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    im = ax.pcolormesh(log_ne, Te_vals, Z, cmap='magma', vmin=0, vmax=1.0, shading='auto')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'$\epsilon_\mathrm{step}$', fontsize=10)
    try:
        cs = ax.contour(log_ne, Te_vals, Z, levels=[0.1, 0.3, 0.5, 0.8],
                        colors='white', linewidths=1.0, alpha=0.8)
        ax.clabel(cs, fmt='%.1f', fontsize=8, inline=True)
    except Exception:
        pass
    add_ITER_box(ax)
    ax.set_xlabel(r'$\log_{10}(n_e\,[\mathrm{cm}^{-3}])$')
    ax.set_ylabel(r'$T_e$ [eV]')
    ax.set_title(r'QSS step sensitivity: initial error after $\Delta T_e=\pm0.6\,\mathrm{eV}$',
                 fontsize=10)
    plt.tight_layout()
    path = f'{out_dir}/fig4_eps_step.png'
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 5 — Population distribution ────────────────────────────────────────
def fig5_populations(out_dir, cr_dir=CR_DIR):
    """n-shell populations vs Saha-Boltzmann at three (Te,ne) reference points."""
    L = np.load(f'{cr_dir}/L_grid.npy')
    S = np.load(f'{cr_dir}/S_grid.npy')

    IH       = 13.6057
    n_shells = np.arange(1, 16)
    n_of_idx = np.array(
        [1] + [2]*2 + [3]*3 + [4]*4 + [5]*5 +
        [6]*6 + [7]*7 + [8]*8 + list(range(9, 16))
    )

    def saha(n_arr, Te_v):
        return 2*n_arr**2 * np.exp(-IH*(1-1/n_arr**2)/Te_v) / 2

    ref_points = [
        (0,  0,  1e14, r'$T_e=1\,\mathrm{eV},\;n_e=10^{12}$',         C['Te1']),
        (np.argmin(np.abs(Te_grid-3)), np.argmin(np.abs(ne_grid-1e14)),
         1e14, r'$T_e=3\,\mathrm{eV},\;n_e=10^{14}$ (ITER)',           C['Te3']),
        (49, 7, 1e14, r'$T_e=10\,\mathrm{eV},\;n_e=10^{15}$',          C['Te10']),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for i_Te, i_ne, n_ion, label, color in ref_points:
        n_ss       = get_ss(L, S, i_Te, i_ne, n_ion)
        n_1S       = max(n_ss[0], 1e-60)
        n_by_shell = np.array([n_ss[n_of_idx == n].sum() / n_1S for n in n_shells])
        s_ref      = saha(n_shells, Te_grid[i_Te])

        ax.semilogy(n_shells, n_by_shell, 'o-', color=color, lw=1.8, ms=5,
                    label=label, zorder=3)
        ax.semilogy(n_shells, s_ref, '--', color=color, lw=1.0, alpha=0.5, zorder=2)

    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0],[0], ls='--', color='gray', lw=1.2, alpha=0.6)]
    labels  += ['Saha-Boltzmann (dashed)']
    ax.legend(handles, labels, fontsize=8.5, loc='lower right', framealpha=0.85)

    ax.set_xlabel(r'Principal quantum number $n$')
    ax.set_ylabel(r'$n(n\text{-shell})\,/\,n(1\mathrm{s})$')
    ax.set_title('Excited-state population distribution vs Saha-Boltzmann', fontsize=10)
    ax.set_xlim(0.5, 15.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.set_ylim(1e-12, 10)
    ax.grid(True, which='both', alpha=0.2, lw=0.5)

    plt.tight_layout()
    path = f'{out_dir}/fig5_populations.png'
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 6 — Regime classification map ──────────────────────────────────────
def fig6_regime_map(data, out_dir):
    """
    Left:  discrete three-regime map (green/yellow/red).
    Right: continuous eps_step heatmap with contour boundaries.
    """
    bd = data['bd']
    Z, Te_vals, log_ne = build_Z(bd, 'eps_step')
    regime = np.where(Z > 0.7, 2, np.where(Z > 0.2, 1, 0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: discrete regime map
    ax = axes[0]
    cmap3 = ListedColormap(['#2ecc71', '#f39c12', '#e74c3c'])
    norm3 = BoundaryNorm([0, 0.5, 1.5, 2.5], cmap3.N)
    im = ax.pcolormesh(log_ne, Te_vals, regime.astype(float),
                       cmap=cmap3, norm=norm3, shading='auto')
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], pad=0.02)
    cb.set_ticklabels([r'QSS valid'+'\n'+r'($\epsilon<0.2$)',
                       'Transitional\n(0.2\u20130.7)',
                       r'Perm. non-QSS'+'\n'+r'($\epsilon>0.7$)'])
    cb.ax.tick_params(labelsize=8)
    try:
        ax.contour(log_ne, Te_vals, Z, levels=[0.2, 0.7],
                   colors=['#27ae60', '#c0392b'], linewidths=1.5, zorder=4)
    except Exception:
        pass
    n_v = (regime==0).sum(); n_t = (regime==1).sum()
    n_p = (regime==2).sum(); tot = regime.size
    ax.text(12.1, 9.5, f'QSS valid: {n_v/tot*100:.0f}%',
            color='#27ae60', fontsize=8, fontweight='bold')
    ax.text(12.1, 8.8, f'Transitional: {n_t/tot*100:.0f}%',
            color='#d35400', fontsize=8, fontweight='bold')
    ax.text(12.1, 8.1, f'Perm. non-QSS: {n_p/tot*100:.0f}%',
            color='#c0392b', fontsize=8, fontweight='bold')
    add_ITER_box(ax)
    ax.set_xlabel(r'$\log_{10}(n_e\,[\mathrm{cm}^{-3}])$')
    ax.set_ylabel(r'$T_e$ [eV]')
    ax.set_title(r'QSS regime classification ($\Delta T_e=\pm0.6\,\mathrm{eV}$)',
                 fontsize=9.5)

    # Right: continuous eps_step
    ax2 = axes[1]
    im2 = ax2.pcolormesh(log_ne, Te_vals, Z, cmap='RdYlGn_r',
                         vmin=0, vmax=1, shading='auto')
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cb2.set_label(r'$\epsilon_\mathrm{step}$', fontsize=10)
    try:
        cs = ax2.contour(log_ne, Te_vals, Z, levels=[0.2, 0.5, 0.7],
                         colors=['#27ae60','#e67e22','#c0392b'],
                         linewidths=1.4, zorder=4)
        ax2.clabel(cs, fmt={0.2:'0.2', 0.5:'0.5', 0.7:'0.7'},
                   fontsize=8, inline=True)
    except Exception:
        pass
    add_ITER_box(ax2)
    ax2.set_xlabel(r'$\log_{10}(n_e\,[\mathrm{cm}^{-3}])$')
    ax2.set_ylabel(r'$T_e$ [eV]')
    ax2.set_title(r'$\epsilon_\mathrm{step}$ (continuous)',  fontsize=9.5)

    plt.tight_layout()
    path = f'{out_dir}/fig6_regime_map.png'
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 7 — eps vs M scatter + Te scaling law ──────────────────────────────
def fig7_eps_scaling(data, out_dir):
    """
    Left:  eps_step vs log(M), coloured by Te — shows NO universal M collapse.
    Right: eps_step vs Te with exponential fit — clean scaling law R²=0.995.

    Key result: eps_step ≈ 1.52 × exp(−0.42 × Te[eV]) + 0.04
    eps_step is driven by Te (Boltzmann gradient), not by M.
    """
    bd  = data['bd']
    M   = data['M']

    # Build flat arrays in Te×ne order matching bd (sorted Te then ne)
    bd_s = bd.sort_values(['Te_eV', 'ne_cm3']).reset_index(drop=True)
    Te_r = bd_s['Te_eV'].values
    ne_r = bd_s['ne_cm3'].values
    eps  = bd_s['eps_step'].values

    M_flat = np.array([
        M[np.argmin(np.abs(Te_grid - Tv)), np.argmin(np.abs(ne_grid - nv))]
        for Tv, nv in zip(Te_r, ne_r)
    ])

    # Fit
    def exp_model(T, a, b, c):
        return a * np.exp(-b * T) + c

    popt, _ = curve_fit(exp_model, Te_r, eps, p0=[0.9, 0.3, 0.05])
    a, b, c = popt
    Te_line = np.linspace(1, 10, 300)
    eps_fit = exp_model(Te_line, *popt)
    R2 = 1 - np.var(eps - exp_model(Te_r, *popt)) / np.var(eps)

    r_M,  _ = pearsonr(np.log10(M_flat), eps)
    r_ne, _ = pearsonr(np.log10(ne_r),   eps)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: eps vs log(M) coloured by Te
    ax = axes[0]
    sc = ax.scatter(np.log10(M_flat), eps, c=Te_r,
                    cmap='plasma_r', s=28, alpha=0.85, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r'$T_e$ [eV]', fontsize=10)
    ax.set_xlabel(r'$\log_{10}(M) = \log_{10}(\tau_\mathrm{QSS}/\tau_\mathrm{relax})$')
    ax.set_ylabel(r'$\epsilon_\mathrm{step}$')
    ax.set_title(r'$\epsilon_\mathrm{step}$ vs memory metric $M$', fontsize=10)
    ax.text(0.05, 0.93, f'Pearson $r = {r_M:.3f}$',
            transform=ax.transAxes, fontsize=9, color='gray')
    ax.text(0.05, 0.84,
            'Stripes = constant $T_e$\n'
            r'$\Rightarrow$ no universal $f(M)$ collapse',
            transform=ax.transAxes, fontsize=8.5, color='#444')

    # Right: eps vs Te with fit
    ax2 = axes[1]
    sc2 = ax2.scatter(Te_r, eps, c=np.log10(ne_r),
                      cmap='viridis', s=28, alpha=0.85, zorder=3)
    cb2 = fig.colorbar(sc2, ax=ax2, pad=0.02)
    cb2.set_label(r'$\log_{10}(n_e\,[\mathrm{cm}^{-3}])$', fontsize=10)
    ax2.plot(Te_line, eps_fit, 'k-', lw=2.5, zorder=4,
             label=fr'${a:.2f}\,e^{{-{b:.2f}\,T_e}}+{c:.2f}$')
    ax2.set_xlabel(r'$T_e$ [eV]')
    ax2.set_ylabel(r'$\epsilon_\mathrm{step}$')
    ax2.set_title(r'$\epsilon_\mathrm{step}$ scaling with $T_e$', fontsize=10)
    ax2.text(0.55, 0.93, f'$R^2 = {R2:.4f}$',
             transform=ax2.transAxes, fontsize=9)
    ax2.text(0.55, 0.84,
             r'$n_e$-independent ($r=' + f'{r_ne:.3f}' + r'$)',
             transform=ax2.transAxes, fontsize=8.5, color='gray')
    ax2.legend(fontsize=9, loc='upper right')

    fig.suptitle(
        r'QSS error scaling: $\epsilon_\mathrm{step}\approx f(T_e)$, not $f(M)$',
        y=1.01, fontsize=11)
    plt.tight_layout()
    path = f'{out_dir}/fig7_eps_scaling.png'
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("GENERATING ALL FIGURES (fig1–fig7)")
    print("=" * 60)
    print(f"  Data:    {DATA_DIR}/")
    print(f"  CR data: {CR_DIR}/")
    print(f"  Output:  {OUT_DIR}/")
    print()

    data = load_data(DATA_DIR)

    paths = [
        fig1_epsilon_traces(data, OUT_DIR),
        fig2_breakdown_map(data, OUT_DIR),
        fig3_timescales(data, OUT_DIR),
        fig4_eps_step(data, OUT_DIR),
        fig5_populations(OUT_DIR, CR_DIR),
        fig6_regime_map(data, OUT_DIR),
        fig7_eps_scaling(data, OUT_DIR),
    ]

    print()
    print("=" * 60)
    print("ALL FIGURES COMPLETE")
    print("=" * 60)
    captions = [
        "Fig 5.1 — Two-stage QSS relaxation (epsilon traces)",
        "Fig 5.2 — QSS breakdown map at ITER timescales",
        "Fig 5.3 — Timescale structure tau_QSS and M",
        "Fig 5.4 — QSS step sensitivity eps_step",
        "Fig 5.5 — Population distribution vs Saha-Boltzmann",
        "Fig 5.6 — Three-regime classification map",
        "Fig 5.7 — eps scaling law: f(Te) not f(M)",
    ]
    for p, cap in zip(paths, captions):
        kb = os.path.getsize(p) / 1024
        print(f"  {cap:<48s}  {kb:.0f} KB")