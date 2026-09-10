---
name: scientific-prose-audit
description: Strict anti-AI-style editor for scientific prose. Makes academic writing read as though written carefully by a competent researcher rather than generated. Use when drafting or revising any thesis chapter, paper, abstract, figure caption, or report; when text must not display stereotypical AI language patterns; and as the final pass after the physics is frozen. Enforces zero em dashes, banned filler transitions, claim strength proportional to evidence, and named physical quantities instead of vague nouns.
---

# Scientific writing anti-AI style auditor

Make academic prose sound like it was written carefully by a technically
competent researcher.

**Do not change scientific meaning, mathematical content, numerical values,
citations, notation, or the strength of a claim unless explicitly asked.**

Priority: precision, natural academic prose, authorial restraint.

---

## Absolute rules

### 1. Never use em dashes

The character `—` must not appear. Replace it with whatever punctuation is
grammatically right: comma, colon, semicolon, parentheses, full stop. Do not
mechanically substitute the same mark every time. Often the cleanest fix is to
delete the interruption and let the sentence run.

Bad: *The excited states relax rapidly—much faster than the ground-state reservoir.*
Better: *The excited states relax much faster than the ground-state reservoir.*

### 2. Banned transitions

It is important to note that · It is worth noting that · It should be noted
that · Interestingly · Notably · Importantly · Crucially · Remarkably · In
essence · At its core · In other words · In this context · In this regard ·
From this perspective · With this in mind · Taken together · All in all ·
Overall, these findings suggest · This highlights · This underscores · This
sheds light on · This provides valuable insight into · This serves as · This
demonstrates the importance of · This offers a framework for · This paves the
way for · A key takeaway is · One may argue that · It becomes evident that

In almost every case, delete the phrase and state the claim.

### 3. No inflated language

Avoid: leverage · utilize (use "use") · facilitate (use "allow") · elucidate ·
delve into · intricate · multifaceted · comprehensive (unless genuinely so) ·
robust (unless robustness was tested) · seamless · pivotal · profound ·
transformative · compelling · novel paradigm · unprecedented · sophisticated ·
holistic · nuanced (unless the nuance is explained) · valuable · insightful ·
groundbreaking · cutting-edge

Do not make ordinary results sound important with adjectives.

### 4. No empty significance statements

Never end a paragraph by telling the reader the result matters. State the
implication itself.

Bad: *This result has significant implications for plasma diagnostics.*
Better: *During this interval, an equilibrium Balmer-ratio table assigns a
temperature to a population state that is not in ionisation equilibrium.*

### 5. No repeated conclusions

The pattern claim → explanation → equation → repeat claim → "This highlights..."
is the signature of generated text. Once the evidence establishes the point,
move on.

### 6. Do not overuse paired contrasts

not only X but also Y · rather than X, Y · not X, but Y · while X, Y ·
although X, Y · both X and Y. Valid English, formulaic when repeated. Keep only
where the contrast carries scientific meaning.

### 7. No artificial rhetorical symmetry

Do not force groups of three, or "First..., second..., finally...", unless the
structure genuinely has parallel parts. Technical writing may be asymmetric.

### 8. No unnecessary signposting

Avoid *In the following section we investigate...*, *We now turn our attention
to...*, *Having established X, we proceed to Y.* Some signposting is useful at
chapter scale, not every few paragraphs.

### 9. No fake reader guidance

Recall that · Notice that · Observe that · One can clearly see · It is easy to
see · It is straightforward to show · As expected · Unsurprisingly.

If something follows mathematically, show why. If it is physically expected,
give the physical reason.

Bad: *As expected, the relaxation time decreases with density.*
Better: *Increasing $n_e$ raises the electron-impact transition frequencies,
which shortens the excited-state relaxation time over this regime.*

### 10. Nothing is obvious

Ban clearly · obviously · evidently · trivially · naturally. These usually hide
a missing argument.

### 11. Name the physical quantity

Avoid this behaviour · this phenomenon · this effect · these dynamics · this
trend · this relationship · this interplay · this mechanism, when the variable
can be named.

Bad: *This behaviour becomes stronger at low temperature.*
Better: *The plateau error increases toward lower $T_e$.*

### 12. Claim strength must match evidence

Distinguish shows · demonstrates · is consistent with · suggests · indicates ·
supports · correlates with · causes. Never substitute causation for
correlation. Never write "proves" for numerical evidence. If a result holds
only over the simulated grid, say so.

Numerical coincidence: *Across the scanned grid, the density of maximum plateau
error coincides with the density of maximum reservoir sensitivity.*
Derived dependence: *Equation (X) identifies the reservoir sensitivity as the
factor carrying this density dependence.*

### 13. Concrete scientific subjects

*The $n=3$ population increases*, not *It can be seen that the population
increases*. Avoid unnecessary passive voice without eliminating it
mechanically; passive is right when the procedure matters more than the actor.

### 14. Do not overuse "we"

Avoid chains of *We calculate... We then determine... We next compare...* Use
the scientific object as subject where natural. Keep "we" for genuine
methodological choices: *We retain $n=1$ as an independent state and apply the
QSS approximation only to the excited manifold.*

### 15. Do not narrate straightforward algebra

Bad: *We can now substitute Eq. (3) into Eq. (4). Upon carrying out this
substitution and simplifying, we obtain...*
Better: *Substitution of Eq. (3) into Eq. (4) gives*

Explain a step only when its reason is not apparent.

### 16. No sentence fragments for effect

*The consequence? A large error.* · *Not equilibrium. Not even close.* These do
not belong in a thesis.

### 17. Few colons. 18. Few parentheses.

If parenthetical information matters to the argument, write it into the
sentence. If it does not, delete it.

### 19. No generic topic sentences

Bad: *Collisional-radiative modelling plays an important role in plasma physics.*
Better: *In non-LTE plasmas, excited-state populations must be obtained from the
balance of collisional and radiative processes.*

### 20. No generic paragraph endings

...which is important for future studies · ...providing deeper insight into the
underlying physics · ...demonstrating the effectiveness of the proposed
approach. End on the scientific statement that matters.

---

## Thesis-specific rules

**21. Build from physics, not terminology.** Introduce a term only after
explaining what physical object or operation it names. Do not write *The
quasi-steady-state approximation is employed*; start from the rate equation and
identify which derivative is neglected and why.

**22. Equations must do work.** Every displayed equation should define a
quantity, derive a result, express a balance, establish a limit, generate a
prediction, or support a later numerical test. Otherwise remove it or move it
to an appendix.

**23. Separate observation from interpretation.** State the numerical
observation first, then interpret. Do not merge them before the evidence is on
the page.

**24. Distinguish definition, derivation, empirical finding.** *We define...*
for definitions. *From Eq. (X)...* for derivations. *The grid scan gives...*
for numerical findings. *This suggests...* for interpretation not yet derived.

**25. Do not retrofit a story to numerical output.** Before calling a numerical
pattern a mechanism, require an analytical derivation, a controlled parameter
variation, an independently tested falsifiable prediction, or a limiting-case
argument. Otherwise call it an observation.

---

## Sentence-level preferences

*The excited manifold relaxes in nanoseconds*, not *It is observed that the
excited manifold undergoes relaxation on a nanosecond timescale.*

*Equation (5.12) separates the operator and reservoir contributions exactly*,
not *Equation (5.12) provides a useful framework through which the respective
contributions can be clearly understood.*

*The two quantities are not equivalent*, not *It is important to emphasize that
these two quantities should not be regarded as being equivalent to one another.*

## Paragraph structure and rhythm

One intellectual unit per paragraph. A useful default: state the question or
claim, give the reasoning or evidence, state the consequence only if it is not
already evident. Do not force this on every paragraph. Vary paragraph length.
Avoid chains of paragraphs with identical length and rhythm, and avoid a
repeated short claim → long explanation → short conclusion cadence.

## Vocabulary repetition

Watch for however · therefore · thus · moreover · furthermore · consequently ·
notably · importantly · specifically · essentially · particularly recurring in
nearby paragraphs. Do not swap in synonyms. Usually the transition can be
deleted.

## Anti-polish rule

Do not make every sentence maximally elegant. Clear before stylish. Preserve
grammatically sound irregularities in the author's voice. Do not rewrite a
simple sentence because a more elaborate version exists.

## Author-voice preservation

Identify the author's normal sentence length, preferred terminology, and level
of formality, and keep them. Correct awkward English without replacing the
author's reasoning. Do not introduce metaphors, add rhetorical emphasis, or add
conclusions the author did not make. The result should sound like the same
researcher writing more carefully, not like a copywriter.

## Scientific integrity rule

Never improve prose by hiding uncertainty. Never remove limitations,
assumptions, exceptions, uncertainty ranges, failed predictions, or
disagreement with the literature because they make a sentence less elegant.
Precision has priority over confidence.

---

## Editing procedure

Run these passes in order.

1. **Meaning lock.** Identify claims, equations, assumptions, numerical values
   and qualifications that must not change.
2. **Remove AI language.** Formulaic transitions, inflated adjectives, generic
   significance statements, rhetorical symmetry, unnecessary summaries.
3. **Simplify.** Direct constructions where no technical meaning is lost.
4. **Scientific precision.** Replace vague nouns and verbs with the actual
   variables, mechanisms, states, equations, or results.
5. **Rhythm.** Break repetitive sentence and paragraph patterns.
6. **Em-dash check.** Search explicitly for `—`. Final count must be zero.
7. **Claim-strength audit.** Check every proves · demonstrates · establishes ·
   causes · determines · validates · confirms against the evidence.
8. **Read as an examiner.** Would any sentence make you ask "How do you know
   that?" If so, supply the reason or weaken the claim.

## Mechanical scan

```bash
# em dashes: must return 0
grep -c '—' FILE

# banned openers and filler
grep -nE 'It is (important|worth) (to )?not|It should be noted|Interestingly|Notably,|Importantly,|Crucially|Remarkably|In essence|At its core|Taken together|This (highlights|underscores|demonstrates the importance|paves the way)' FILE

# inflated vocabulary
grep -noE '\b(leverage|utilize|facilitate|elucidate|delve|intricate|multifaceted|seamless|pivotal|profound|transformative|compelling|unprecedented|holistic|groundbreaking|cutting-edge)\b' FILE

# unearned certainty
grep -noE '\b(clearly|obviously|evidently|trivially|naturally|as expected|unsurprisingly)\b' FILE

# vague subjects
grep -noE '\bthis (behaviour|behavior|phenomenon|effect|trend|interplay|relationship)\b' FILE
```

## Required output when auditing

**A. Revised text.** The complete edited passage.

**B. Problems removed.** Only meaningful ones: AI filler, repeated argument,
unsupported claim strength, vague physical subject, unnecessary signposting,
inflated language, em dash, awkward mathematical narration. Not trivial copy
edits.

**C. Scientific changes.** Either "No scientific meaning changed." or an
explicit list of every scientific change proposed. Never silently modify
scientific content.

## Final acceptance test

1. Zero em dashes?
2. Could any sentence have appeared unchanged in thousands of unrelated
   AI-generated papers?
3. Does every adjective earn its place?
4. Does every paragraph advance the argument?
5. Are physical quantities named rather than called "this effect"?
6. Are claims no stronger than the evidence?
7. Is any conclusion repeated after being demonstrated?
8. Does it sound like a researcher explaining their own calculation?
9. Would deleting a sentence lose scientific information?
10. Is the prose simpler than the version received, unless more explanation was
    genuinely needed?

If 1 is no, reject. If several of 2 to 10 are no, revise again.

## Ordering constraint

This is a **final** pass. Run it after the physics is frozen. Good prose is not
evidence, and editing text whose claims are still moving wastes the work. If the
physics changes during editing, go backwards through the earlier review gates
rather than repairing it with prose.
