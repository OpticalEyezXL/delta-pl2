# Δe_n = n²e_n — 계산 검증 코드

Optical Eyez XL, *Integration of the Generation–Trace Structure of Fundamental
Physical Laws II* (vol2) 논문에서 사용된 계산 검증 스크립트 모음.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `vol2_verification.py` | 논문 부록(Appendix A–Y)의 주요 계산을 sympy/numpy로 검증하는 스크립트. 네 극한, 랭크-2 조석장, 슈바르츠실트 일치, Kerr 고리특이점, SU(3) 닫힘, SU(2) 이중덮개 등을 포함. |
| `vol2_verification_output.txt` | 위 스크립트의 실행 결과 로그. |
| `quantum_phase_sum_gated.py` | §3.4의 위상합 S(τ)=Σcₙe^(in²τ)를 양자회로(Qiskit)로 구현한 예비실험. 이상적 시뮬레이터 버전과, 실제 위상게이트(P)·제어위상게이트(CP)로 분해한 버전 둘 다 포함. 게이트 수가 큐비트 수에 대해 다항식(O(q²))으로 증가함을 확인. |

## 실행 방법

### `vol2_verification.py`

```bash
pip install sympy numpy scipy
python vol2_verification.py
```

### `quantum_phase_sum_gated.py`

```bash
pip install qiskit
python quantum_phase_sum_gated.py
```

## 참고

- 모든 계산은 답을 미리 정해놓고 확인하는 방식이 아니라, 입력(생성자 Δ, 조건식)에서
  시작해 실제로 미분·전개·수치계산을 거쳐 결과를 얻는 방식으로 작성되었다.
- 외부의 알려진 사실(예: 슈바르츠실트 조석장의 표준값, 파울리 행렬)과 대조하는
  부분은 코드 주석에 `[외부 사실, 하드코딩]`으로 명시해 구분해 두었다.
- 양자회로의 게이트 분해 기법(이진전개 기반 이차형태 위상 인코딩)은 양자 산술회로
  문헌에서 이미 표준적으로 쓰이는 구성이며, 이 저장소가 새로 고안한 기법이 아니다.
