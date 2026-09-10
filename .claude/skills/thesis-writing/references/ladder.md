# Developing an idea from first principles

The ladder below is for material the reader has not met. Standard undergraduate
mathematics does not need it and climbing the ladder for something familiar is
condescension dressed as rigour.

## The ladder

**1. Physical question.** State what has to be determined. *The population of an
excited state changes because particles enter and leave it through competing
processes, so the first task is its population balance.* Do not open with a
matrix equation that nothing has yet motivated.

**2. Governing principle.** Name the law it descends from: particle
conservation, energy conservation, probability balance, detailed balance,
reaction kinetics, linear response, radiative transition probability.

**3. Simplest mathematical form.** The smallest equation that carries the idea.

**4. Define every new object** — symbol, units, indices, signs, state space,
dimensionality, and what kind of quantity it is. Define at the point of need,
not in a nomenclature block the reader must hold in memory until it becomes
relevant.

**5. Add processes one at a time.** Show individual terms entering before
compressing into vector or matrix notation. The compressed form is easy to read
once you know what it contains and opaque before.

**6. State assumptions where they enter,** with what they remove. Maxwellian
electrons, fixed electron density, optically thin plasma, quasineutrality,
isolated atom, neglected molecular channels, frozen variables, bundled levels,
linearisation. An assumption introduced ten pages after its first use is a
finding waiting to happen.

**7. Generalise** only now: matrix notation, arbitrary state count, block
decomposition, eigenmode expansion, operator form, numerical algorithm.

**8. Check.** Dimensions, signs, conservation, symmetry, detailed balance,
equilibrium, and limits — zero density, high density, short time, long time,
vanishing perturbation, known analytic result, independently tabulated value. A
limiting case that comes out right teaches more than another paragraph of
explanation, and one that comes out wrong is worth more than either.

**9. Interpret.** What has the equation taught us physically? Translating the
symbols back into words is not interpretation. *The off-diagonal elements
transfer population between states while the diagonal collects each state's
total loss rate* is.

**10. Connect.** Why this result is needed for the research question.

## Mathematics is argument, not decoration

Before an important equation, say why it is being introduced. After it, say only
what is not already visible in it.

Show enough steps that a capable Master's student can reconstruct the derivation
without guessing. Skip routine algebra that adds nothing. The author's
familiarity with a step is not evidence that the reader shares it.

Never silently replace an exact expression with an approximation. Give the exact
relation, the approximation, the condition it requires, and what it costs
physically.

Keep notation fixed. Changing a symbol for prose variety is a defect, not
elegance.

Treat every equation as a claim: where it came from, when it is valid, what
would break it.

## Explaining a mechanism

For each central mechanism, eventually answer all four:

- **What happens?** The phenomenon.
- **Why?** The mechanism.
- **What sets its size?** Governing parameters and scalings.
- **When does it matter?** The regime.

*The error increases at lower temperature* is an observation. The explanation
begins at the next question: which rate, population channel, timescale or
coupling changes with temperature in a way that produces this? Follow the causal
chain until it terminates in established physics or in an explicitly stated
unresolved mechanism. Stopping earlier and calling it an explanation is the most
common way a results chapter fails.

**Depth test.** Could the reader reproduce this explanation without memorising
the sentence? If not, it is description wearing an explanation's clothes.

## Alternative explanations

Before attributing an observed pattern to a mechanism, ask what else produces
it: numerical artifact, grid resolution, step-size definition, interpolation,
normalisation choice, boundary condition, omitted process, finite-state
truncation, incorrect scaling, diagnostic degeneracy. Test what can be tested,
name what cannot. A mechanism is established when the alternatives have been
addressed, not when it is the first explanation that fit.

## Complexity

Do not simplify by deleting physics. Simplify by controlling when information
arrives — motivation first, notation staged, assumptions local, every
transformation purposeful, consequence stated. Complexity is acceptable.
Unstructured complexity is not.
