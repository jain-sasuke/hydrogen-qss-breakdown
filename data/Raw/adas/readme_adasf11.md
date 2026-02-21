
**Understanding ADAS ADF11 Master Files (scd96 / acd96)**  
Project: Hydrogen Collisional–Radiative (CR) Modeling Thesis

---

# 1. Purpose

This document clarifies the structure of ADAS ADF11 master files and documents a critical parsing confusion that was resolved through reference to the official ADAS manual.

This ensures:

- Correct unit interpretation
- Correct grid indexing
- Correct reshaping of coefficient arrays
- Physics-consistent usage in CR modeling

---

# 2. The Initial Confusion

When parsing `scd96_h.dat` and `acd96_h.dat`, the numeric grids appeared as:

```
7.69897 ... 15.30103
-0.69897 ... 4.00000
```

Initial incorrect interpretation:

- First grid = log10(Te [K])
- Second grid = log10(ne [cm^-3])

This led to:

- Te ≈ 4300 eV → 1.7×10^11 eV
- ne ≈ 0.2 → 10^4 cm^-3

This contradicted the Open-ADAS website metadata.

This interpretation was incorrect.

---

# 3. Official ADF11 Specification

According to the ADAS manual (Appendix A11 – ADF11 format):

Header structure:

```
IZMAX, IDMAXD, ITMAXD, IZ1MIN, IZ1MAX
```

Grids are stored in the following order:

```
(DDENSD(ID), ID=1,IDMAXD)
(DTEVD(IT), IT=1,ITMAXD)
```

Where:

- DDENSD = log10(electron density [cm^-3])
- DTEVD  = log10(electron temperature [eV])

Coefficient storage:

```
For IT = 1, ITMAXD
   ( DRCOFD(*,IT,ID), ID=1,IDMAXD )
```

Meaning:

- Outer loop → temperature (IT)
- Inner loop → density (ID)

---

# 4. Correct Interpretation of Our File

Header line:

```
1   24   29   1   1
```

Mapping:

- IDMAXD = 24 → 24 density points
- ITMAXD = 29 → 29 temperature points

Therefore:

Grid 1 (24 numbers):
→ log10(ne [cm^-3])

Grid 2 (29 numbers):
→ log10(Te [eV])

Converted ranges:

Density range:

10^7.69897 → 10^15.30103  
≈ 5×10^7 → 2×10^15 cm^-3

Temperature range:

10^-0.69897 → 10^4  
≈ 0.2 eV → 10^4 eV

This matches the Open-ADAS metadata exactly.

---

# 5. Final Data Structure

Coefficient array shape:

```
(ITMAXD, IDMAXD)
= (29 temperatures, 24 densities)
```

Mathematically:

K(Te, ne)

NOT:

- K(n, Te)
- K(level, Te)
- K(n,l,Te)

ADF11 contains effective collisional–radiative coefficients only.

---

# 6. Physical Meaning of Density Dependence

Although ionization is fundamentally a two-body process, ADF11 coefficients are effective CR rates.

They include:

- Collisional redistribution
- Metastable coupling
- Three-body recombination
- Density-dependent excited state populations

Therefore:

K_eff = K(Te, ne)

not just K(Te).

---

# 7. Correct Parsing Logic

Minimal correct Python parsing:

```python
logne = nums[:IDMAXD]
logTe = nums[IDMAXD:IDMAXD + ITMAXD]
logK  = nums[IDMAXD + ITMAXD:]

logK = logK.reshape((ITMAXD, IDMAXD))

ne_cm3 = 10**logne
Te_eV  = 10**logTe
K_cm3s = 10**logK
```

No transpose required.

---

# 8. Key Lessons

1. Never assume units from magnitude.
2. Always consult the official ADAS manual.
3. ADF11 is charge-state resolved, not level-resolved.
4. Grids are stored in log10.
5. Temperature is stored in eV (not Kelvin).
6. Density grid appears first in file.

---

# 9. Final Verified Structure

```
Header: IZ, IDMAXD, ITMAXD, ...

Grid 1: log10(ne [cm^-3])  → 24 values
Grid 2: log10(Te [eV])     → 29 values

Coefficients:
For each temperature IT:
    24 density values
```

Final shape:

```
K.shape = (29, 24)
```

