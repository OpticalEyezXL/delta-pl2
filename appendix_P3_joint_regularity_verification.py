"""
§5.4 결합계 정칙성 계층 k*_joint = min(k*_1, k*_2) 검증 (정확한 수렴판정)

Tr(Delta^k C_0) = sum_n n^{2k-2*alpha}  (p-급수)
수렴 조건: 2k - 2*alpha < -1  <=>  alpha > k + 1/2   (표준 p-급수 판정, 절단근사 아님)

Tr(Delta_joint^k C_0_joint) = sum_{j=0}^k C(k,j) * Tr(Delta^j C_0^(1)) * Tr(Delta^(k-j) C_0^(2))
전체 합이 유한 <=> 모든 j=0..k 항이 유한 <=> (j=0항, j=k항이 가장 발산하기 쉬우므로) 그 둘이 유한
<=> alpha_1 > k+1/2 이고 alpha_2 > k+1/2
"""
import sympy as sp
import math

n = sp.symbols('n', positive=True, integer=True)

def p_series_converges(exponent_value):
    """sum_{n=1}^inf n^{exponent} 의 수렴 여부를 sympy로 정확히 판정 (절단 없음)."""
    s = sp.Sum(n**exponent_value, (n, 1, sp.oo))
    return bool(s.is_convergent())

def k_star(alpha):
    return math.floor(alpha - 0.5 - 1e-12)

def check_case(name, alpha1, alpha2, k_range):
    print(f"=== {name}: alpha1={alpha1}, alpha2={alpha2} ===")
    k1s, k2s = k_star(alpha1), k_star(alpha2)
    predicted = min(k1s, k2s)
    print(f"  k*_1={k1s}, k*_2={k2s}  ->  예측 k*_joint = min = {predicted}")
    for k in k_range:
        # j=0항: Tr(Delta^0 C0^1)*Tr(Delta^k C0^2) -> 발산 여부는 두 번째 인자(지수 2k-2a2)가 결정
        conv_j0_second = p_series_converges(2*k - 2*alpha2)
        conv_j0_first  = p_series_converges(0 - 2*alpha1)
        term_j0_finite = conv_j0_second and conv_j0_first
        # j=k항: Tr(Delta^k C0^1)*Tr(Delta^0 C0^2)
        conv_jk_first  = p_series_converges(2*k - 2*alpha1)
        conv_jk_second = p_series_converges(0 - 2*alpha2)
        term_jk_finite = conv_jk_first and conv_jk_second
        overall_finite = term_j0_finite and term_jk_finite
        should_finite = (k <= predicted)
        status = "OK" if overall_finite == should_finite else "MISMATCH"
        print(f"  k={k}: j=0항 유한={term_j0_finite}, j=k항 유한={term_jk_finite} "
              f"-> 전체유한={overall_finite}, 예측={should_finite} [{status}]")
    print()

check_case("비대칭 낮음",  alpha1=1.2, alpha2=2.4, k_range=range(0,3))
check_case("대칭 높음",    alpha1=5.0, alpha2=5.0, k_range=range(3,6))
check_case("극단 비대칭",  alpha1=0.6, alpha2=8.0, k_range=range(0,3))
