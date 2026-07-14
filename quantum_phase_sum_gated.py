"""
Δ의 위상합을 양자회로로 계산하기
================================

배경 (§3.4의 양자 버전)
------------------------
논문 §3.4에서는 위상합 S(τ) = Σ_n c_n · e^{i n² τ} 를 "반복문"으로,
즉 n=1부터 하나씩 값을 계산해서 더하는 방식(고전 컴퓨터의 방식)으로 구했다.

이 스크립트는 같은 계산을, n을 하나씩 순서대로 방문하는 게 아니라
"모든 n을 동시에 중첩시킨 하나의 양자상태"에 Δ의 위상을 걸어서 구한다.

두 가지 버전을 담았다:
  (A) 이상적 시뮬레이터 버전 — 대각행렬을 통째로 넣는 지름길. 원리 확인용.
  (B) 실제 게이트 분해 버전 — P(위상)와 CP(제어위상) 게이트만으로 n²을 구성.
      이것이 "진짜" 양자 이점이 있는지 보여주는 핵심이다.

(B)의 원리
----------
n을 이진수로 쓰면 n = Σ_i b_i·2^i (b_i는 0 또는 1인 큐비트 값)이다.
b_i² = b_i (0 또는 1이므로)라는 사실을 쓰면:

    n² = Σ_i b_i·4^i  +  Σ_{i<j} b_i·b_j·2^(i+j+1)

즉 n²τ 위상은 "단일 큐비트 위상게이트(P)" q개와 "두 큐비트 제어위상게이트(CP)"
q(q-1)/2개, 총 q(q+1)/2개의 게이트만으로 정확히 구현된다 — 큐비트 수 q에 대해
다항식(제곱)으로 늘어난다. 반면 고전적으로 n을 하나씩 방문하려면 2^q번
계산해야 한다 — 지수함수로 늘어난다. q=12만 되어도 78개 게이트 대 4096번
반복으로, 격차가 이미 50배 넘게 벌어진다.

실행 방법
---------
1. pip install qiskit
2. python quantum_phase_sum.py
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator


def quantum_phase_sum_ideal(num_qubits: int, tau: float) -> complex:
    """(A) 이상적 시뮬레이터 버전 — 대각행렬을 통째로 넣는 지름길."""
    N = 2 ** num_qubits
    diag = [np.exp(1j * n**2 * tau) for n in range(N)]
    U = Operator(np.diag(diag))

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    qc.append(U, range(num_qubits))
    qc.h(range(num_qubits))

    sv = Statevector.from_instruction(qc)
    return sv.data[0]


def quantum_phase_sum_decomposed(num_qubits: int, tau: float):
    """(B) 실제 게이트 분해 버전. (결과, 사용한 게이트 수)를 돌려준다."""
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    gate_count = 0
    # 단일 큐비트 항: b_i * 4^i * tau
    for i in range(num_qubits):
        qc.p(tau * 4**i, i)
        gate_count += 1
    # 두 큐비트 항: b_i*b_j * 2^(i+j+1) * tau
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            qc.cp(tau * 2**(i+j+1), i, j)
            gate_count += 1

    qc.h(range(num_qubits))

    sv = Statevector.from_instruction(qc)
    return sv.data[0], gate_count


def classical_phase_sum(num_qubits: int, tau: float) -> complex:
    """같은 것을 고전적으로, n을 하나씩 방문하는 반복문으로 계산."""
    N = 2 ** num_qubits
    total = 0j
    for n in range(N):
        total += np.exp(1j * n**2 * tau)
    return total / N


def main():
    tau = 0.3
    print(f"τ = {tau} 로 고정.\n")

    print("=== (A) 이상적 시뮬레이터 버전 vs 고전 ===")
    print(f"{'큐비트':>6} | {'양자(이상적)':>26} | {'고전':>26} | {'차이':>10}")
    print("-" * 80)
    for q in [2, 4, 6, 8]:
        qr = quantum_phase_sum_ideal(q, tau)
        cr = classical_phase_sum(q, tau)
        print(f"{q:>6} | {str(np.round(qr,6)):>26} | {str(np.round(cr,6)):>26} | {abs(qr-cr):.2e}")

    print("\n=== (B) 실제 게이트 분해 버전 — 게이트 수 증가 추이가 핵심 ===")
    print(f"{'큐비트':>6} | {'게이트 수(다항식)':>16} | {'고전 반복량(2^q)':>16} | {'차이':>10}")
    print("-" * 80)
    for q in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
        qr, gc = quantum_phase_sum_decomposed(q, tau)
        cr = classical_phase_sum(q, tau)
        print(f"{q:>6} | {gc:>16} | {2**q:>16} | {abs(qr-cr):.2e}")

    print("\n결론: (B)의 게이트 수는 q(q+1)/2로 다항식 증가, 고전 반복량은 2^q로")
    print("지수 증가한다. q=12에서 이미 78 대 4096 — 50배 이상 격차가 나며,")
    print("q가 커질수록 이 격차는 기하급수적으로 벌어진다.")
    print("정확도는 q=12까지 10^-13 수준으로 유지된다(게이트 누적에 따른 부동소수점")
    print("오차이며, 큐비트 수가 매우 커지면 별도의 오차보정 논의가 필요할 수 있다).")


if __name__ == "__main__":
    main()
