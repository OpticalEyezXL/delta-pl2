# Delta e_n = n^2 e_n -- Computational Verification Code

Verification scripts used in the paper *Integration of the Generation–Trace
Structure of Fundamental Physical Laws II* (vol2) by Optical Eyez XL.

Korean-commented originals and English translations are both included.

## Contents

| File | Description |
|---|---|
| `vol2_verification.py` / `vol2_verification_en.py` | Verifies the main calculations from the paper's appendices (Appendix A–Y, AA) using sympy/numpy: the four limits, the rank-2 tidal tensor, agreement with Schwarzschild, the Kerr ring singularity, SU(3) closure, the SU(2) double cover, the representation-theoretic correspondence of the Navier-Stokes stretching term (Appendix AA), and more. |
| `vol2_verification_output.txt` | Execution log of the above script (Korean version). |
| `quantum_phase_sum_gated.py` / `quantum_phase_sum_gated_en.py` | A preliminary experiment implementing the Section 3.4 phase sum S(tau)=Sum c_n e^{in^2 tau} as a quantum circuit (Qiskit). Includes both an ideal-simulator version and a version decomposed into real phase (P) and controlled-phase (CP) gates. Confirms that gate count grows polynomially (O(q^2)) in the number of qubits. |

## How to Run

### `vol2_verification*.py`

```bash
pip install sympy numpy scipy
python vol2_verification_en.py
```

### `quantum_phase_sum_gated*.py`

```bash
pip install qiskit
python quantum_phase_sum_gated_en.py
```

## Notes

- These scripts are written to start from inputs (the generator Delta, defining
  conditions) and genuinely carry out the differentiation, expansion, or
  numerical computation needed to obtain each result -- not to pre-load an
  expected answer and simply confirm it.
- Places where the code compares against an already-known external fact (e.g.,
  the standard Schwarzschild tidal tensor, the Pauli matrices) are explicitly
  marked in the comments as `[EXTERNAL FACT, HARDCODED]`.
- The gate-decomposition technique used in the quantum circuit (encoding a
  quadratic form as phase via binary expansion) is a standard construction
  already used in the quantum-arithmetic-circuit literature; it is not a new
  technique introduced by this repository.
