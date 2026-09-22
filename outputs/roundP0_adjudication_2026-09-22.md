# Round P0 (provenance and regeneration kill test): adjudication, 22 September 2026

Adjudicated with the review-audit skill against the live tree. Three read-only
subagents traced each claim from the thesis sentence back to the producing
artifact before the reviewer's reasoning was read, and the reviewer's own
provenance PASS verdicts were audited at the same standard, which is where the
largest finding came from.

The reviewer states at the outset that they audited the PDF only and could not
inspect the repository. That is the right disclosure to make, and it explains
the shape of the result below: the claims that can be settled from the PDF are
largely right, and the claims about what the repository does are largely not.

## Verdict table

| # | Claim (restated) | Class | Verdict | Check that settled it |
|---|---|---|---|---|
| P0.1 | The three timescale arrays have two writers, and this contaminates M = 9982, the M map, the location of the M maximum, the factor 19.3, plateau-window membership and the transport calculation | provenance | **Partially correct**: defect real, contamination refuted on all six | Each of the six was traced to its artifact and producer. None reads the contested arrays. M = 9982 is tau_slow/tau_relax computed from L_grid by `verify_timescales.py` into `validation/spectrum_testpoint.csv`; the 19.3 comes from `m_rank_test` via `divertor_map.csv`; window membership is recomputed from `eigvals(L)` inside the producing script, not read from a file; `verify_transport_selection.py` takes tau_QSS from `divertor_map.csv`. The thesis has no M-map figure; its only M figure is drawn from `plateau_gridmap.csv`. Further, the two writers are not two calculations: both sort the eigenvalues of the same canonical L and keep the negatives, neither applies the historical filter, and the arrays on disk agree with the independent artifact to 2e-16. |
| P0.2 | The two abstract-level QSS errors, 8.66e-6 and 6.73e-9, are sourced to a working note rather than a script | provenance | **Correct, and understated** | Both values sit in `validation/trajectory_census/trajectory_census.csv`, a stamped artifact, as `max_track_w30_shell` at [23,5] and [0,4]: 8.65971e-6 and 6.72618e-9, with the paired CRE values 0.0633859 and 0.386902. Understated in two ways: the citation was the last surviving internal file name in the whole tree, which today's prose cleanup missed, and the reason the values looked unstamped is a naming gap, the artifact calling the quantity `max_track` where the thesis calls it eps_QSS. |
| P0.3 | Table 5.4 carries the crest mechanism but only two of its four columns have a stated producer | provenance | **Correct, and understated** | Every one of the 20 printed values reproduces from stamped artifacts, so the table is not unscripted. But reproducing them exposed something the reviewer did not see: the printed ln(u_CRE/u_peak) takes its numerator at temperature index 23 and its denominator at index 24, a single logarithm of a ratio evaluated at two different temperatures, and the caption said the table was at index 24. Evaluating it consistently at either index shifts every entry by about 0.25 and moves neither the sign change nor the maximum. |
| P0.4 | Table 5.2's caption is stale; the values are stamped | provenance | **Correct** | All nine G values reproduce from `reservoir_gain.csv` to six digits. The body two paragraphs below the table already said so, contradicting its own caption. |
| P0.5 | Table 4.4 says four severe checks rest on notes, but the rows show none | presentation | **Correct** | At HEAD exactly one severe row was in the notes condition while the caption said four, so the sentence was already wrong by three; stamping that row earlier today made it zero. All five other counts in the caption check out. |
| P0.6 | The writer audit cannot certify what it is used for | tooling | **Correct, already disclosed** | Chapter 4 already states the defect and the 45 call sites, and says the audit should not be cited as evidence. |
| P0.7 | Headline provenance table, mostly PASS | evidence | **Several unearned clearances** | See below. |
| P0.8 | Other numbers still rest on working notes | provenance | **Correct** | 33 provenance sites in live prose: 23 support current claims, 7 are historical records of superseded values with their stamped replacement named, 3 are definitional. Of the 23, eight are pure re-citations, seven are short reductions of existing artifacts, and eight would need a run. |

## What the review got wrong, and why it matters

**The contamination claim does not survive.** The reviewer's severity rests on
the dual-writer defect reaching load-bearing numbers. It reaches none of the six
they name. The honest answer to their examiner question is not the defensive one
they suggest. It is: the abstract's M was never read from an M array at all.

**The Declaration is quoted with its opening clause removed.** The review quotes
"every numerical result is produced by a script in the accompanying repository
and quoted with the script and the data file that produced it". The Declaration
reads "The standard it is written to is that every numerical result is produced
by a script...". The elision turns a stated standard into a claim of universal
achievement, which makes the conflict look sharper than it is. The substance,
that some numbers did not meet the standard, was still right.

## The largest finding, from auditing the reviewer's PASS verdicts

The reviewer marked several results PASS on provenance, among them the transport
condition failing at 0 of 45 pairs and the joint-step worst case of 22.1 per
cent. Those verdicts are unearned, and not for the reason they would guess.

**24 of the 97 CSV files under `validation/` carry no header at all**: no
producing script, no interpreter, no input hash. Among them are the artifacts
behind the reservoir gain, the transport selection, the joint-step map, the
emissivity generalisation, the fault injection, the crest subgrid, the operator
conditioning and the Lyman trapping census. The producing scripts print the
input hashes to standard output, so the provenance existed at run time; it was
simply never written into the file. An artifact with no header cannot answer the
question the reviewer is asking, whoever wrote it.

This is a broader defect than the dual-writer one the review led with, it
affects abstract-level numbers, and the review could not have found it from the
PDF. Writing those headers means re-running the producers, so it is the author's
call and nothing was regenerated here.

## What was done

**`verify_headline_provenance.py`** → `validation/headline_provenance/`. Reads
the stamped artifacts and re-derives all 45 printed numbers that carried a
working-note citation: the two QSS errors and their CRE pairs, the nine gains of
Table 5.2, the twelve entries of Table 4.2, and the twenty entries of Table 5.4.
All 45 reproduce to the precision printed. It computes no new physics; every
value is a reduction of an artifact produced by another script, and it names
that script and column for each. It also classifies each source artifact's own
provenance and reports, rather than hides, that one of the four is unstamped.

**`verify_ladder_counts.py`**. Parses Table 4.4's rows and checks every count its
caption states, exiting non-zero on drift. It independently reproduced the five
correct counts and caught the sixth.

**Five re-citations applied**: Table 5.2, Table 4.2 and Table 5.4 captions, the
chapter 5 sentence carrying the two QSS errors, and the abstract. The Table 5.4
caption now states the mixed-index convention explicitly and the size of the
alternative. The Table 4.2 caption now says which eigensolver the printed column
is. Chapter 4's two-writer disclosure now carries the two facts that bound it:
the writers are byte-identical because they run identical code on the same input,
and no thesis number reads them.

## What is worth the author's time

**Author's decisions, not mine.** Writing provenance headers into the 24
unstamped artifacts, which means re-running their producers; giving the three
timescale arrays a single writer; and the eight working-note numbers that would
need a run, of which the load-bearing ones are the ratio |lambda_0| to
K_ion n_e, the eigenvalue conditioning and single-precision test, and the
ell-weighting invariance of Delta.

**Ignore.** The contamination claim; the demand to rerun every downstream
consumer of the timescale arrays, which have no thesis-facing consumers; and the
instruction to retitle the section on unrepaired defects, since both defects are
still unrepaired and saying so is the stronger position.

## What the review missed

The 24 unstamped artifacts, above. Also, from the same inventory: the
cap-utilisation percentages quoted as 51 and 97 per cent rest on caps printed as
0.4503 and 0.4394 where the stamped artifacts give 0.451357 and 0.441707, a
third-digit disagreement; and the crest vertex quoted as 3.66e13 differs from
`crest_subgrid`'s 3.7047e13. Neither changes a conclusion, both are live
EVIDENCE numbers, and neither was raised.

## Source calibration

Eight substantive claims: four correct, two correct and understated, one
partially correct with its severity refuted, one a set of PASS verdicts that do
not hold. No fabrications, and every number it quoted from the PDF was accurate.
Characteristic failure mode: it reasons confidently about repository structure
from a document that describes it, so its factual claims about what the text says
are reliable and its claims about what the code does are not. It also
over-escalates: a latent hazard that touches no printed number is filed as MAJOR
alongside two real unscripted tables. Its strengths are worth keeping: it found
the Table 4.4 contradiction, which is a genuine ledger defect, and it was right
that two abstract-level numbers and a mechanism table needed stamping. Read its
document-level claims directly; re-derive anything it says about the repository.
