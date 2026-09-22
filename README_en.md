# Delta e_n = n^2 e_n -- Computational Verification Code

Verification scripts used in the paper *Integration of the Generation–Trace
Structure of Fundamental Physical Laws II* (vol2) by Optical Eyez XL.

Korean-commented originals and English translations are both included.

## Contents

| File | Description |
|---|---|
| `vol2_verification.py` / `vol2_verification_en.py` | Verifies the main calculations from the paper's appendices (Appendix A–Y, AA) using sympy/numpy: the four limits, the rank-2 tidal tensor, agreement with Schwarzschild, the Kerr ring singularity, SU(3) closure, the SU(2) double cover, the joint regularity hierarchy k*_joint=min(k*_1,k*_2) (Appendix P.3), the representation-theoretic correspondence of the Navier-Stokes stretching term (Appendix AA), and more. |
| `vol2_verification_output.txt` | Execution log of the above script (Korean version). |

## How to Run

### `vol2_verification*.py`

```bash
pip install sympy numpy scipy
python vol2_verification_en.py
```

## Notes

- These scripts are written to start from inputs (the generator Delta, defining
  conditions) and genuinely carry out the differentiation, expansion, or
  numerical computation needed to obtain each result -- not to pre-load an
  expected answer and simply confirm it.
- Places where the code compares against an already-known external fact (e.g.,
  the standard Schwarzschild tidal tensor, the Pauli matrices) are explicitly
  marked in the comments as `[EXTERNAL FACT, HARDCODED]`.
