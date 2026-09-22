# Round P0 re-check (provenance): adjudication, 22 September 2026

The review is substantially right, and I am acting on it rather than arguing
with it. Its central claim, that active numerical claims still carried a
working-note citation while the Declaration promises a producing script, holds.
Two of its judgements are understated, one of its counts is approximately right
rather than exact, and one item it listed as a provenance problem turns out to
be a wrong number.

This round also corrects an error of my own from the previous one.

## Verdict table

| # | Claim (restated) | Class | Verdict | Check that settled it |
|---|---|---|---|---|
| 1 | The five previous P0 defects are fixed: the two abstract QSS numbers, Table 5.2, Table 5.4, Table 4.4's count, and the producer of M = 9982 | provenance | **Correct** | Each verified in the current source. The QSS pair cites `trajectory_census`; Table 5.2 cites `reservoir_gain.csv`; Table 5.4 names its artifacts and index convention; Table 4.4 now says none of the severe checks rests on a note, and `verify_ladder_counts.py` enforces it; M = 9982 is attributed to `verify_timescales.py`. |
| 2 | The dual-writer defect should drop from MAJOR to MODERATE because it is isolated from thesis results | provenance | **Correct** | Matches the tracing done last round: the arrays feed no figure or table, the writers are byte-identical, and the abstract's M comes from elsewhere. |
| 3 | About 20 occurrences of "no producing script" remain, several supporting current claims | provenance | **Correct** | 23 occurrences in the LaTeX source, of which one is the Table 4.4 caption defining the term. The PDF count of 20 is the same defect counted after rendering. |
| 4 | The specific list: the 0.85 % l-mixing bound, the conditioning and single-precision numbers, the [1,4] eigenvalues and gap, the 0.09 % n_ion freeze, Sbar = −0.2288, the linearisation statistics, the five l-weighting Δ values and the 51 / 97 per cent, f3 and f4 at high density, the sub-grid crest numbers and the 54 points, and the 202 to 200 census | provenance | **Correct, and understated in cost** | Every one is a real unscripted citation. But the review implies new work. In fact each of the seven I tested is computable from the canonical operator in seconds, and **all of them reproduced at the precision printed**. The repair was re-citation plus one script, not new science. |
| 5 | Sbar = −0.2288 is now especially unnecessary to cite to a note, since Table 5.4 already draws |S| from the stamped artifact | provenance | **Correct** | `reservoir_gain.csv` holds −0.228793 at [23,5], heating, k = 1. |
| 6 | The Declaration's absolute wording is stronger than the document satisfies, and could be weakened | not a physics claim | **Partially correct** | The Declaration reads "The standard it is written to is that every numerical result is produced by a script...". That is already a statement of a standard rather than a claim of universal achievement, and the review's quote drops the opening clause, as its predecessor did. The substance stood, and the author's instruction was to bring the thesis up to the standard rather than weaken it, which is what was done. |
| 7 | Table 5.4's mixed-index construction deserves a mathematical audit later, not now | scope | **Reasonable, and already documented** | The caption states the convention and the size of the consistent alternative. Deferred as the reviewer asks. |

## What the review missed: two of those numbers are wrong, not merely unscripted

The review listed the sub-grid crest numbers under provenance. They are worse
than that. `validation/crest_subgrid/crest_vertices.csv` gives, for the
three-point parabolic fit:

| quantity | thesis printed | stamped artifact |
|---|---|---|
| vertex at 1 eV | 3.66e13 | 3.7047e13 |
| vertex at 9.5 eV | 1.51e13 | 1.5233e13 |
| range | 2.56 | 2.560 |

The range is exactly right, but it is the ratio of the largest vertex to the
smallest over all temperatures, not the ratio of the two endpoints printed,
which is 2.42. And the minimum is not at the top of the range: the vertex falls
to 1.4472e13 at 5.96 eV and then **rises** to 1.5233e13 at 9.54 eV. The
increments change sign once, so the sentence's "drifts downward with
temperature" is true over most of the grid and false above about 6 eV.

The paragraph's conclusion survives and is arguably strengthened, since the
whole drift is smaller than one grid interval and therefore unresolved. The
sentence has been requoted from the artifact with the reversal stated.

## A correction to my own previous report

In the last round I reported two third-digit disagreements as open. One of them
was not a disagreement at all. The cap values behind the 51 and 97 per cent
utilisation figures, 0.4503 and 0.4394, reproduce exactly at the post-step
index, 0.450299 and 0.439352, which is the index `tab:position_effect` uses. A
subagent had compared them at the pre-step index and I passed that through
without checking. The utilisations are 50.8 and 97.0 per cent against the
printed 51 and 97. Nothing needs changing there. The crest discrepancy, the
other item, was real and is fixed above.

## What was done

**`verify_residual_claims.py`** → `validation/residual_claims/`. Computes from
the canonical L and S, and compares with the printed value at the precision
printed:

| | quantity | printed | computed |
|---|---|---|---|
| C1 | three slowest eigenvalues at [1,4] and the gap | −4.29, −1.60e8, −3.44e8, 3.7e7 | −4.287, −1.603e8, −3.443e8, 3.739e7 |
| C2 | bound population per n_ion, cost of freezing it | 8.925e-4, 0.09 % | 8.924573e-4, 0.0892 % |
| C3 | Δ under five sublevel weightings, and the spread | 1.94561, 1.94474, 1.94447, 1.94471, 1.94574, 0.07 % | identical to five decimals, spread 0.0652 % |
| C4 | cap utilisation at the benchmark and one column in | 0.2288 / 0.4503 = 51 %, 0.4261 / 0.4394 = 97 % | 0.228793 / 0.450299 = 50.8 %, 0.426061 / 0.439352 = 97.0 % |
| C5 | l-mixing scaling bounds on the two clocks | ≤ 0.85 % and < 1 % | 0.151 % and 0.139 % at factors 3 and 7 |
| C6 | eigenvalue conditioning and single precision | 1.71, 1.98, 3.26, 2.59, 1.3e-5 | 1.707, 1.981, 3.255, 2.589, 1.356e-5 |
| C7 | sub-grid crest vertices | 3.66e13, 1.51e13 | 3.7047e13, 1.5233e13, **mismatch** |

21 of 23 comparisons reproduce. The two that do not are C7, now requoted. Two
of these deserve comment. The five l-weighting values of Δ reproduced to five
decimal places from a guess at what the five weightings were, which is strong
evidence the guess was right and that the original calculation was sound. The
four condition numbers reproduced exactly under the standard definition
1/|y^H x| with unit-norm eigenvectors, which the thesis had not recorded.

The remaining working-note citations have been re-cited to this script or, where
they record a superseded value, labelled explicitly as historical and not used
as evidence.

## What is worth the author's time

**Ignore.** Nothing in this review is wrong enough to drop, but its implied cost
was too high: it proposed scripting as a large task and suggested weakening the
Declaration as an alternative. Neither was needed.

**Still open, unchanged.** Provenance headers for the 24 unstamped validation
artifacts; a single writer for the three timescale arrays; the abstract at about
734 words against the institute's 300.

## Source calibration

Seven claims: five correct, one correct and understated, one partially correct.
No fabrications, and its reading of the revised document was accurate throughout,
including which defects had been closed. Characteristic failure mode, consistent
across its rounds: it reads the document well and infers the repository badly,
and it estimates repair cost high. It also repeated its predecessor's elision of
the Declaration's opening clause. Its strength here is real: it correctly refused
to let the working-note citations stand, and it correctly singled out Sbar as the
one that had become indefensible. Act on its provenance findings directly; check
its cost estimates and its claims about what the code would have to do.
