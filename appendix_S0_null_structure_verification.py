"""
Appendix S.0 — 실수축·회전축 누적량의 null 구조 검증
vol2 §8.2 관련

A(t) := -log(Tr(C_t)/Tr(C_0)) = integral_0^t R(t') dt'  (실수축 누적량)
R(t) -> 2 (t -> inf) 이므로 A(t) ~ 2t + const 로 점근한다는 것을 직접 이산합으로 확인한다.
"""
import numpy as np

def trace_Ct(t, alpha=2.0, N=200000):
    n = np.arange(1, N + 1)
    return np.sum(n ** (-2 * alpha) * np.exp(-2 * n ** 2 * t))

def R_numeric(t, alpha=2.0, h=1e-6, N=200000):
    f1 = trace_Ct(t + h, alpha, N)
    f0 = trace_Ct(t - h, alpha, N)
    return -(np.log(f1) - np.log(f0)) / (2 * h)

def main():
    alpha = 2.0
    ts = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    Tr0 = trace_Ct(1e-6, alpha)

    print("=== S.0. A(t) = -log(Tr(C_t)/Tr(C_0)) 의 점근적 선형성 ===")
    print(f"{'t':>6} | {'R(t)':>10} | {'A(t)':>14} | {'A(t)-2t':>12} | {'A(t)/t':>10}")
    consts = []
    for t in ts:
        r = R_numeric(t, alpha)
        Tt = trace_Ct(t, alpha)
        A = -np.log(Tt / Tr0)
        const = A - 2 * t
        if t >= 1.0:
            consts.append(const)
        print(f"{t:6.2f} | {r:10.5f} | {A:14.6f} | {const:12.5f} | {A/t:10.5f}")

    # t>=1 구간에서 A(t)-2t가 상수로 수렴하는지 확인 (표준편차가 아주 작아야 함)
    consts = np.array(consts)
    spread = consts.max() - consts.min()
    print(f"\nt>=1 구간에서 A(t)-2t 값들의 최대-최소 차: {spread:.2e}")
    assert spread < 1e-3, "A(t)-2t가 상수로 수렴하지 않음"
    print("-> A(t) ~ 2t + const, 점근적 선형성 확인됨 (기울기 2 = R(inf))")

    # 지배모드 근사: Phi(t) ~ t (n=1), 비율 c = A(t)/Phi(t) -> 2
    print("\n=== 지배모드 궤적의 null 비율 c = A(t)/Phi(t) -> R(inf) = 2 ===")
    for t in [10.0, 50.0, 100.0]:
        A = -np.log(trace_Ct(t, alpha) / Tr0)
        Phi = t  # n=1 dominant mode
        c = A / Phi
        print(f"  t={t:6.1f}: c = A(t)/Phi(t) = {c:.6f}")

if __name__ == "__main__":
    main()
