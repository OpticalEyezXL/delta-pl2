# Δe_n = n²e_n — 계산 검증 코드

Optical Eyez XL, *Integration of the Generation–Trace Structure of Fundamental
Physical Laws II* (vol2) 논문에서 사용된 계산 검증 스크립트 모음.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `vol2_verification.py` | 논문 부록(Appendix A–Y, AA)의 주요 계산을 sympy/numpy로 검증하는 스크립트. 네 극한, 랭크-2 조석장, 슈바르츠실트 일치, Kerr 고리특이점, SU(3) 닫힘, SU(2) 이중덮개, 실수축·회전축 누적량의 null 구조(Appendix S.0), 결합계 정칙성 계층 k*_joint=min(k*_1,k*_2)(Appendix P.3), 나비에-스토크스 신장항의 표현론적 대응(Appendix AA) 등을 포함. |
| `vol2_verification_output.txt` | 위 스크립트의 실행 결과 로그. |

## 실행 방법

### `vol2_verification.py`

```bash
pip install sympy numpy scipy
python vol2_verification.py
```

## 참고

- 모든 계산은 답을 미리 정해놓고 확인하는 방식이 아니라, 입력(생성자 Δ, 조건식)에서
  시작해 실제로 미분·전개·수치계산을 거쳐 결과를 얻는 방식으로 작성되었다.
- 외부의 알려진 사실(예: 슈바르츠실트 조석장의 표준값, 파울리 행렬)과 대조하는
  부분은 코드 주석에 `[외부 사실, 하드코딩]`으로 명시해 구분해 두었다.
