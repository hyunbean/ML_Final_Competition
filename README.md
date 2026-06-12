# KML Challenge 2026S — 현대백화점 고객 성별 예측

거래 로그를 custid 단위로 집계해 **고객 성별(이진, ROC-AUC)** 을 예측한 팀 프로젝트.
**최종: Public 1위 → Private 2위 (shake-up).** 팀: 김민형·신현빈·조아인·이두영 (hyunbean).

## 📊 결과 요약
| | Public | Private | 순위 |
|---|---|---|---|
| 우리 팀 | 0.7385141 (1위) | 0.7324252 | **2위** |
| 박진성(우승) | 0.7378695 | 0.7345426 | 1위 |

- Public 1위였으나 private에서 −0.00212로 역전패 = **public-overfit**.
- 패인·교훈 전체 분석은 **[`KML2026S_회고_1위.md`](KML2026S_회고_1위.md) 섹션 17~19** 참조 (이 repo에서 가장 중요한 문서).

## 📁 구조
```
KML2026S_회고_1위.md   ★ 회고 (협업·돌파·검증·앙상블구조·shake-up분석·다음대회청사진)
실험노트_템플릿.md      일일 실험노트 템플릿
PIPELINE.md            파이프라인 개요
src/                   핵심 코드 (아래 참조)
experiments_archive/   실패/폐기 실험 코드 (회고 근거용 보관, 재현 비대상)
artifacts/oof/         팀 공유 base 멤버 OOF+test 예측 (스태킹 입력)
artifacts/             train/test custids, folds (고정 seed)
hyunbin_npy/           신현빈 생성 base 멤버 예측
mega피처_김민형/       김민형 mega-FE 자산
```

## 🔑 핵심 파이프라인 (src/)
| 파일 | 역할 |
|---|---|
| `config.py` | 경로·시드(42)·CV·컬럼 정의 (모든 모듈의 기준) |
| `data.py`, `folds.py` | 데이터 로드 / 고정 5-fold (StratifiedKFold seed42) |
| `oof_io.py` | OOF·test 예측 저장/로드 (팀 공유 포맷, 97곳에서 사용) |
| `features.py`, `train_first.py` | 521 집계 FE 생성 (`build_all`) |
| `train_pseudo_strict.py` | pseudo-labeling student (teacher 격리, fold↑ variance 레버). 환경변수 `PL_KFOLD`/`PL_NOSTOP` 등 |
| `blend_rank.py` | rank 가중 블렌딩 (`rank(Σwᵢ·rank(srcᵢ))`) |
| `stack_make73.py`, `blend_stack3.py` | 메타스택 (#73 계열) |
| `train_mega_*.py`, `train_first_*.py` | base 멤버 생성 (GBDT/AG/interact 등) |

## ⚙️ 재현 메모
- 데이터(`*.csv`)는 git 제외 — 각 머신에 배치 후 `KML_DATA` 경로 지정.
- 모든 예측은 **rank 정규화 + 고정 folds + custid 정렬 통일** (스태킹 누수 방지).
- pseudo OOF는 즉석 K-fold라 5-fold 메타스택 입력 금지 (단독/단순 rank-avg만).
- 환경 설정: `COLAB_SETUP.md` / `DLPC_SETUP.md` 참조.

## 💡 다음 대회 핵심 교훈 (회고 섹션 19 — 범용 원칙)
1. **CV−LB Gap 모니터링** — 일정=robust, 벌어지면 과적합. CV 최고 ≠ 채택.
2. **final은 public-max 금지** + 저상관 독립 헤지 확보.
3. **adversarial validation으로 분포차 먼저** — 분포 다르면 robust 처방(drift 제한·균등rank·pseudo 회피), 같으면 공격적으로.
4. **새 FE(진짜 신호) > 가중치 튜닝**, 제출 수 통제.
