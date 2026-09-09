# Correction to `findings_09_central_quantity_misnamed.md` — grid indices in W1 and W2

**Raised 24 August 2026.** Source: `verify_ch3_claims.py` and
`verify_ch3_groupB.py`, both run against
`L_grid.npy` SHA-256
`2d92b58e1693107d9ef8097a778ddec003db6cd1c0aa700762aa0680d059224e`
on a 50 × 8 = 400-point grid.

Two grid indices in the withdrawal table disagree with a direct recomputation.
This matters more than an ordinary typo would, because W1–W6 exist to stop
withdrawn claims being re-quoted, and a withdrawal record that points at the
wrong grid point cannot do that job.

---

## W2 — `[1,0]` should be `[0,0]`

Current text:

> $M_{\max} = 1.73\times10^9$ at [1,0], $n_e = 10^{12}$

Measured:

```
all-grid M   86.77 at [49,7] (Te=10 eV, ne=1e+15)
          .. 1.729e9 at [0,0] (Te=1 eV, ne=1e+12)
```

The $n_e$ agrees; the $T_e$ index is one too high. $T_e[0] = 1$ eV exactly,
which is the value the run prints alongside the index, so the point is pinned
independently of the index convention.

**Suggested replacement:** `at [1,0]` → `at [0,0], $T_e = 1$ eV`.

## W1 — `heat[48,7]` disagrees with the measured location of $M_{\min}$

Current text:

> $M = 86.77$ is at heat[48,7], which has `window_ok = False` and **is not in
> the analysis**

Measured: $M_{\min} = 86.7678$ at `[49,7]`, $T_e = 10$ eV,
$n_e = 10^{15}$ cm⁻³ — the last index in both directions. The *value* matches
to five figures; the index does not.

**Do not patch this one blindly.** The two discrepancies run in opposite
directions — W2 is one too high in $T_e$, W1 one too low — so this is not a
single off-by-one convention. Two possibilities, and they need different fixes:

1. `heat[i,j]` indexes a **(point, direction) list** rather than the
   $(T_e, n_e)$ grid, in which case the index is correct in its own frame and
   only needs its frame stated.
2. It is a transcription error, in which case it should read `[49,7]`.

Resolve by finding the script that produced the `heat`/`cool` lists and
checking what its first index runs over.

---

## What survives unchanged

Everything else in W1–W6 that the recomputation touches:

| quantity | findings_09 | measured |
|---|---|---|
| $M_{\min}$ | 86.77 | 86.7678 |
| $M_{\max}$ | $1.73\times10^9$ | $1.729\times10^9$ |
| $\tau_{\rm QSS}$ max | 67.2 s | 67.2333 s |
| grid size | 400 | 400 (50 × 8) |
| `window_ok` $\equiv M > 900$ | — | 346 of 400 points pass |

W1's substantive point is confirmed: the point carrying $M_{\min}$ has
$M = 86.77 < 900$, so it fails `window_ok` and is outside the analysed set.
Only the coordinate is in question, not the argument.

## Related, and separately verified

§5.3 of findings_09 gives the current ranges as "$M$ from 86.8 to
$1.73\times10^9$, $\tau_{\rm QSS}$ up to 67.2 s". All three reproduce. The
$\tau_{\rm QSS}$ **floor** does not: `chapter3.tex` carried
$\tau_{\rm QSS} \ge 1.18\ \mu$s, which is the minimum over the 346 points with
$M > 900$, not over the 400. Over all 400 it is 75.4 ns, at the same `[49,7]`
corner. Corrected in place in §3.3.4 of the chapter.
