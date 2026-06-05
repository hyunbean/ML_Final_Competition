# 📦 세션1 → 팀원 공유: mega(572) 피처 행렬 + TE 표시

> 요청대로 mega 피처행렬을 custid 인덱스로 export. 본인 376피처와 합쳐 새 모델 학습용.
> custid 정렬 = `train/test_transactions.csv`의 `custid.unique()` (세션2 블록·base와 동일). 누수 0(아래 조건).

## 파일
| 파일 | 내용 |
|---|---|
| `mega_train.parquet` | (30000, 573) = **custid** + f0~f571 |
| `mega_test.parquet` | (19995, 573) = **custid** + f0~f571 |
| `mega_folds.csv` | custid, fold(0~4), gender — **TE 안전용 동일 폴드** |
| `mega_columns_manifest.csv` | f-index, block, **is_target_encoded**, note |

합치기: `df = my376.merge(mega_train, on='custid')` (test 동일).

## ⚠️ 타깃인코딩(TE) 컬럼 = 36개 — fold 안 맞으면 누수

**TE 컬럼 (성별 사용, fold 의존적):**
- `f219~f238` (20개): `te_{corner,pc,brd,buyer,part}_{mean,wmean,max,min}` — 카테고리 성별 타깃인코딩
- `f545~f560` (16개): B2 성별 log-odds (친화도/conflict/gap)

**나머지 536개 = target-free → 그냥 합치면 됨.**

### TE를 안전하게 쓰는 두 가지 방법 (택1)
1. **(권장·제로코스트) 동일 폴드 재사용** — 이 TE는 **seed42 5fold StratifiedKFold(gender 층화)** 로 fold-safe 계산됨(corr=1.0 검증). `mega_folds.csv`의 fold를 **그대로 써서** CV하면 36개 포함 **전부 안전**. 별도 처리 불필요.
   ```python
   folds = pd.read_csv('mega_folds.csv')              # custid, fold
   # 본인 CV에서 이 fold 컬럼으로 split → TE 누수 없음
   ```
2. **다른 폴드를 써야 하면** — TE 36개(`is_target_encoded==True`)를 **드롭**하고 본인 폴드로 **재계산**. (원본 TE는 우리 폴드 기준이라, 다른 폴드에선 검증고객 정답이 섞임 = 이두영 케이스: CV↑ LB추락.)
   ```python
   te_cols = man.loc[man.is_target_encoded, 'col']    # 36개
   X = mega_train.drop(columns=te_cols)               # target-free 536만 사용
   ```

## block 구성 (manifest 참고)
| f-index | block | TE? |
|---|---|---|
| f0–f410 | base411 (수제집계 + W2V) | f219–238만 TE |
| f411–f430 | v4_extra (RFM·시간·바스켓·가격·엔트로피) | — |
| f431–f478 | W2V brd(48) | — |
| f479–f526 | W2V good(48) | — |
| f527–f544 | B1 가격/할인/할부/수입 | — |
| **f545–f560** | **B2 성별 log-odds** | **TE 16** |
| f561–f566 | B3 충성/반복 | — |
| f567–f571 | B4 시간대(점심=직장인/오전=주부) | — |

## 단일 성능 참고 (OOF, seed42 5fold)
- mega 572 단독 xgb ≈ 0.707 / AutoGluon best_quality ≈ 0.718
- megaA(621), s2_pool_sel(360, importance+corr 선택) = AG 0.722 도 있음 — 필요하면 동일 포맷으로 추가 export 가능.

---
*문의: 세션1. 다른 셋(megaA/s2_pool_sel) 또는 컬럼 의미 상세 필요하면 요청.*
