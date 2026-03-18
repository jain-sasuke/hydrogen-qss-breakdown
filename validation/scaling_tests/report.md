# Decoupling of Amplitude and Timescale in Collisional-Radiative QSS Breakdown

## 1. Introduction

Collisional-radiative (CR) models are widely used to describe non-equilibrium plasmas, particularly in fusion-relevant environments such as tokamak edges and divertors. A common simplification is the quasi-steady-state (QSS) approximation, where excited-state populations are assumed to instantaneously adapt to changing plasma conditions.

However, during transient events (e.g. ELM crashes, detachment transitions), this assumption can break down. While it is known that QSS validity depends on characteristic timescales, a quantitative and predictive understanding of **QSS breakdown amplitude and dynamics** remains incomplete.

This work demonstrates that:

> **QSS breakdown is governed by two fundamentally distinct mechanisms:**
>
> - **Amplitude of deviation** is controlled primarily by electron temperature \(T_e\)
> - **Relaxation dynamics (memory)** are controlled by timescale separation \(M = \tau_{\text{QSS}} / \tau_{\text{relax}}\)

---

## 2. Collisional-Radiative Model

The evolution of level populations \( \mathbf{n} \) is governed by:

\[
\frac{d\mathbf{n}}{dt} = \mathbf{C}(T_e, n_e)\, \mathbf{n}
\]

where:
- \( \mathbf{C} \) is the CR rate matrix
- Includes electron-impact excitation/ionization and radiative decay
- Depends on electron temperature \(T_e\) and density \(n_e\)

At steady state:

\[
\mathbf{C}(T_e, n_e)\, \mathbf{n}_{\text{QSS}} = 0
\]

---

## 3. Perturbation by Temperature Step

We consider a sudden perturbation:

\[
T_e \rightarrow T_e + \Delta T_e
\]

The CR matrix changes:

\[
\mathbf{C}(T_e + \Delta T_e)
= \mathbf{C}(T_e)
+ \Delta T_e \frac{\partial \mathbf{C}}{\partial T_e}
+ \mathcal{O}(\Delta T_e^2)
\]

Immediately after the perturbation, populations remain at the old QSS:

\[
\mathbf{n}(t=0^+) = \mathbf{n}_{\text{QSS}}(T_e)
\]

The mismatch generates a transient deviation.

---

## 4. Definition of QSS Breakdown Error

We define the instantaneous QSS error:

\[
\epsilon_{\text{step}} = \left\| \mathbf{n}(t=0^+) - \mathbf{n}_{\text{QSS}}(T_e + \Delta T_e) \right\|
\]

This measures how far the system is from the new steady state immediately after perturbation.

---

## 5. First-Order Scaling Analysis

Using linear perturbation:

\[
\delta \mathbf{n} \sim
\left( \frac{\partial \mathbf{C}}{\partial T_e} \right) \mathbf{n} \cdot \Delta T_e
\]

Thus:

\[
\epsilon_{\text{step}} \propto
\left\| \frac{\partial \mathbf{C}}{\partial T_e} \mathbf{n} \right\| \Delta T_e
\]

---

## 6. Temperature Dependence of Rates

Electron-impact processes dominate CR kinetics. These rates scale approximately as:

\[
K(T_e) \sim T_e^{-1/2} \exp\left(-\frac{E}{T_e}\right)
\]

(see e.g. Fujimoto, *Plasma Spectroscopy*, 2004)

Taking derivative:

\[
\frac{\partial K}{\partial T_e}
\sim
\left( \frac{E}{T_e^2} + \frac{1}{2T_e} \right)
\exp\left(-\frac{E}{T_e}\right)
\]

Dominant term:

\[
\frac{\partial K}{\partial T_e}
\sim \frac{E}{T_e^2} \exp\left(-\frac{E}{T_e}\right)
\]

---

## 7. Resulting Scaling Law

Substituting:

\[
\epsilon_{\text{step}} \sim
\Delta T_e \cdot \frac{E_{\text{eff}}}{T_e^2} \exp\left(-\frac{E_{\text{eff}}}{T_e}\right)
\]

This implies:

### Key result:

\[
\boxed{
\epsilon_{\text{step}} \sim \Delta T_e \cdot e^{-E_{\text{eff}}/T_e}
}
\]

---

## 8. Physical Interpretation

### 8.1 Exponential Dependence

- At low \(T_e\): excitation strongly suppressed → large sensitivity → large ε  
- At high \(T_e\): rates saturate → system closer to equilibrium → small ε  

---

### 8.2 Breakdown of Linear Scaling

At low \(T_e\), threshold effects dominate:

\[
K \approx 0 \quad \Rightarrow \quad \text{nonlinear response}
\]

Thus:

\[
\epsilon \neq \Delta T_e \cdot f(T_e)
\]

---

### 8.3 Density Independence

Rates scale as:

\[
\text{Rate} \sim n_e K(T_e)
\]

But QSS populations satisfy:

\[
\mathbf{C}(T_e, n_e)\, \mathbf{n}_{\text{QSS}} = 0
\]

Leading-order dependence cancels, yielding:

\[
\boxed{
\epsilon_{\text{step}} \approx f(T_e), \quad \text{weakly dependent on } n_e
}
\]

---

## 9. Timescale Separation and Memory

Define:

\[
M = \frac{\tau_{\text{QSS}}}{\tau_{\text{relax}}}
\]

where:
- \( \tau_{\text{QSS}} \): fast equilibration timescale
- \( \tau_{\text{relax}} \): slow manifold timescale

Eigenvalue decomposition:

\[
\mathbf{C} = \mathbf{V} \Lambda \mathbf{V}^{-1}
\]

gives:

\[
\tau_i = \frac{1}{|\lambda_i|}
\]

---

### Key observation:

\[
\boxed{
M \text{ controls relaxation dynamics, not amplitude}
}
\]

---

## 10. Decoupling Principle

We obtain:

| Quantity | Governing parameter |
|--------|--------------------|
| Error amplitude \( \epsilon_{\text{step}} \) | \(T_e\) |
| Relaxation time | \(M\) |
| Density dependence | weak for amplitude |

---

### Final statement:

\[
\boxed{
\text{QSS breakdown amplitude and dynamics are governed by distinct physical mechanisms}
}
\]

---

## 11. Implications

### 11.1 Fusion plasmas

- ELMs (µs timescale): large ε → QSS invalid  
- Detachment (ms timescale): system relaxes → QSS valid  

---

### 11.2 Spectroscopy

Non-equilibrium populations lead to:

\[
I_{ij} \propto n_i A_{ij}
\]

Thus QSS error → emission error.

---

### 11.3 Modeling

QSS validity cannot be assessed by timescale alone; both:

- amplitude (Te-driven)
- relaxation (M-driven)

must be considered.

---

## 12. Assumptions

1. Optically thin plasma  
2. Maxwellian electron distribution  
3. Hydrogen-like system  
4. Small-to-moderate ΔTe perturbations  
5. No external particle sources  

---

## 13. Limitations

- Multi-element plasmas may introduce additional coupling  
- Strong radiation trapping not included  
- High-density regimes (LTE limit) not explored  

---

## 14. References

1. Fujimoto, T. *Plasma Spectroscopy*, Oxford (2004)  
2. Griem, H. R. *Principles of Plasma Spectroscopy* (1997)  
3. Ralchenko, Y. *Modern Methods in CR Modeling* (2016)  
4. Sobelman, I. *Atomic Spectra and Radiative Transitions*  

---

## 15. Conclusion

This work demonstrates that QSS breakdown is not governed by a single universal parameter. Instead, a **two-parameter structure emerges**:

- \(T_e\): controls deviation amplitude  
- \(M\): controls relaxation and memory  

This decoupling provides a new framework for understanding and predicting non-equilibrium effects in transient plasmas.
