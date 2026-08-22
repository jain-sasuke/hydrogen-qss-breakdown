---
name: thesis-verification
description: The verification methodology for this CR-modelling thesis — how to establish that a computational result means what it claims before it enters the thesis. Use when producing or checking any number, auditing a script, resolving conflicting values across documents, promoting a backlog entry, or deciding whether a claim is supported. Covers the six-step rhythm, the ground-truth hierarchy, the ten error-reducing methods, and the backlog graduation states.
---

# Thesis verification methodology

Turning results you *have* into results you can *defend*.

## Why this exists

Scientific code produces numbers. Numbers enter documents. Documents get cited
by other documents. Somewhere in that chain a structurally-consistent but
physically wrong result can enter and propagate — passing every automated check,
appearing in every summary, surviving until someone derives it by hand.

This project has a −46% Hα result that did exactly that. It passed conservation,
sign, and NaN checks. It was still an artifact, traceable to a spurious
eigenmode in a stale matrix.

The defence is an unbroken chain from physical principle to published number.

## Ground-truth hierarchy

**physics → math → code → documents**

Physics beats math: if the algebra contradicts a limit, the algebra is wrong.
Math beats code: if code disagrees with a correct derivation, the code is wrong.
Code beats documents: if a note quotes a number the code does not produce, the
note is stale.

**Documents are never authorities** — not the master plan, not the backlog, not
CLAUDE.md's reference values, not a previous conclusion. They are consistency
checks. A document does not become true by being recent.

## The six-step rhythm

One quantity per session:

1. **Physics** — the picture in words, from primary sources (Fujimoto, Griem,
   van der Mullen, Badnell), never from project notes.
2. **Hand derivation** — the student derives on paper. Ask and wait. Deriving it
   for them produces recognition, not understanding, and recognition collapses
   under defense questioning.
3. **Worked example** — real numbers to an actual value. Symbolic work hides
   unit errors and inverted ratios; arithmetic exposes them.
4. **Brutal physics test** — limits, signs, dimensions, magnitudes,
   conservation. Hostile on purpose.
5. **Code** — only now, line by line, against the hand derivation.
6. **Note** — written before moving on.

## Derive at the rigour of the code you audit

If the code implements a full published formula with angular factors and cutoffs
and the derivation is an order-of-magnitude estimate, a disagreement is
ambiguous: code bug, or expected toy-versus-full gap? Resolving that ambiguity
costs more than deriving properly would have. This happened with the ℓ-mixing
audit.

## Ten methods that reduce error

Each of these caught something real in this project.

**1. Predict before you compute.** Write the expected answer down first. This
converts a computation into a test. A number produced with no prior expectation
can only be rationalised, never falsified.

**2. State what would refute it.** Name the observation that would kill the
claim. If nothing would, it is not empirical.

**3. Severity, not consistency.** A test a wrong result would also have passed
tells you nothing. Ask: would this check have caught the error if present?

**4. Sensitivity to definitions.** Test under the alternatives. If the sign
flips, it is an artifact. If only the magnitude moves, refuse to quote a precise
value. (The boundary descent survived three weightings — hence no fitted
exponent is quoted.)

**5. Disagreement is the signal; agreement is weak.** Two language models
agreeing may share a training bias. Their disagreement locates real uncertainty.
Feed each the raw output, never the other's interpretation.

**6. Provenance on every number.** "τ_relax = 2.28 ns" is not a result.
"τ_relax = 2.277 ns from eig(L_grid[23,5]), regenerated 14 Jul after the F(U_m)
fix, reproduced on my machine" is.

**7. Record corrections in place.** "An earlier draft claimed X; this was an
arithmetic error; corrected to Y; the conclusion that depended on it is
retracted." A silently fixed note lets the same wrong path be re-derived later.

**8. Units on everything.** Wrong dimensions means wrong formula, no further
checking needed. Three separate units errors have occurred here.

**9. Convergence before belief.** For any stiff integration, tighten tolerances
and shrink max_step. Stable answer means physics; drifting answer means
numerics.

**10. Separate the claim from the observable.** A result can be true in one
measure and false in another. Every claim states its norm and its perturbation.

## Auditing code

Read the source, not the docstring. Docstrings drift and often record original
intent rather than current behaviour. A comment saying "we neglect X because it
is small" is a claim to test — check whether X is actually small under the
conditions in use. (In this project such a comment concealed a factor of 3–7.)

When code deviates from the formula it cites, trace to the primary source and
answer three questions:

1. Is the deviation real? (Sometimes the docstring is wrong and the code right.)
2. How large is it? Quantify. "Drops a factor of 3–7 here" is actionable;
   "looks wrong" is not.
3. Does it matter for the headline result? Perturb by the size of the error and
   measure the effect.

That third question changes the character of a finding. "I found a bug, traced
it to the source equation, quantified it, and proved the reported result is
insensitive to it" is evidence of rigour. Finding a bug and not bounding it just
creates doubt.

## Backlog graduation

💡 Idea → 📐 Derived → 💻 Coded → ▶️ Run → ✅ Verified

To reach ✅: reproduced on the student's own machine · sensitivity-checked
against definitional choices · caveats written down · a named thesis home · a
stated falsifier that did not occur.

**Nothing enters the thesis before ✅.** A result that runs but is not understood
is a black box, and a black box produced the −46% artifact.

Add ideas the moment they surface — one line is enough. Do not stop to derive.

## Writing results up

Structure per derivation note: core result (boxed) · physical picture readable
by someone a year junior · full derivation · worked example with real numbers ·
checks passed · honest limitations · what remains open · defense one-liners.

Keep each display equation on one line inside `$$...$$` with blank lines around
it. Multi-line equations inside the delimiters fail to render in most markdown
viewers — a silent failure where the derivation is correct but unreadable.

## Two failure modes to watch

**Derivation before intuition** produces symbol-shuffling: the algebra can be
manipulated but no term means anything.

**Code before numerical example** produces black boxes: the script runs, the
numbers look plausible, and nobody can tell whether they are right.
