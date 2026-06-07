# 📦 세션1 피처 시도본 전체 (block_s1_*) — 강한 base 검증 결과

> custid 정렬 = train/test `custid.unique()` (mega/s2_pool_sel과 동일, fold5/seed42). 전부 fold-safe.
> ⚠️ **핵심 교훈**: mega(약한 base) 증분은 착시. **s2_pool_sel(강한 base, 3시드) Δ가 진짜.**

## 강한 base(s2_pool_sel 0.71366 ±0.00053) 위 순증분
| 블록 | 피처수 | 내용 | mega Δ (착시) | **s2_pool_sel Δ (진짜)** |
|---|--:|---|--:|--:|
| **interact** | 108 | top25 피처 2-way 곱 + null-importance 선택 | — | **+0.00075** ⚠️선택과적합 의심 |
| cross | 6 | 취향 부호역전(선물): 습관 반대방향 구매 | +0.0011 | **−0.00131** |
| brandext | 5 | 외부 브랜드→타깃성별 (174 인식브랜드) | +0.0023 | **−0.00077** |
| ext | 11 | 외부 corner/pc 타깃성별 + deconf gap | +0.0012 | **−0.00130** |
| text | 48 | 상품명 ko-SBERT→SVD48 | (−) | **−0.00166** |

## 해석
- **cross/brandext/ext/text = 강한 base에서 전부 음성** (내부 TE가 이미 흡수, 중복→과적합). mega 증분은 약한-base 착시였음.
- **interact(+0.00075) = 강한 base에서 유일하게 양수** — GBT가 자동으로 못 잡은 명시적 2-way 상호작용이 약간 있음. **단 ⚠️ null-importance 선택을 전체 train에 해서 선택 과적합 가능성** → nested-CV 확인 전엔 신뢰 보류. 전체 300개(미선택)는 −0.0002라 **선택이 관건.**

## 사용법
```python
import pandas as pd
base = pd.read_parquet('../mega_train.parquet')   # 또는 s2_pool_sel
interact = pd.read_parquet('block_s1_interact_tr.parquet')
X = pd.concat([base, interact], axis=1)   # custid 순서 동일
```
- TE 포함 블록 없음(cross/brandext/ext/text/interact 전부 target-free 또는 fold-safe). mega의 TE컬럼(f219-238,f545-560)과 별개.
- ⚠️ 권장: **interact만 검토 가치**(나머지 4종은 강한 base에서 음성 확정 → 미사용).

## 재현 스크립트 (~/Desktop/dacon final competition/)
- interact: `_fe_interact.py` / cross: `build_cross_block.py` / brandext: `proto_brand_external.py` / ext: `build_external_block.py` / text: `build_text_member.py`

---
*세션1 2026-06-06. interact +0.00075는 nested-CV 확인 필요(선택과적합 의심). 나머지는 강한 base 음성 확정.*


## 모델 OOF 시도본 (model_oof_attempts/) — 직교 멤버 후보
> 전부 fold5/seed42 OOF. 앙상블 기여는 거의 0(프론티어 redundant 또는 약함). 기록·재현용.

| 멤버 | OOF AUC | 비고 |
|---|--:|---|
| augdeep_mlp_adv | 0.71481 | 증강 딥(adv/manifold 등) |
| augdeep_mlp_manifold | 0.71432 | 증강 딥(adv/manifold 등) |
| augdeep_mlp_swapdae | 0.71360 | 증강 딥(adv/manifold 등) |
| augdeep_mlp_vime | 0.71350 | 증강 딥(adv/manifold 등) |
| augdeep_mlp_cutmix | 0.71278 | 증강 딥(adv/manifold 등) |
| augdeep_mlp_smote | 0.70625 | 증강 딥(adv/manifold 등) |
| rank_s2_pool_sel_xgb_rank | 0.69975 | AUC-direct 랭킹(목적함수) |
| rank_mega_xgb_rank | 0.69423 | AUC-direct 랭킹(목적함수) |
| text_ngram_char | 0.66934 | 텍스트(SBERT/n-gram/Set-T) |
| text_settransformer | 0.66214 | 텍스트(SBERT/n-gram/Set-T) |
| rank_s2_pool_sel_lgb_rank | 0.66152 | AUC-direct 랭킹(목적함수) |
| text_sbert_svd48 | 0.65118 | 텍스트(SBERT/n-gram/Set-T) |
| transduct_s2_pool_sel_labelspread | 0.64446 | transductive 라벨전파 |
| rank_mega_lgb_rank | 0.62721 | AUC-direct 랭킹(목적함수) |

→ 전부 GBT/AG(0.714~0.722) 아래 + corr 높거나 약함 = **앙상블 기여 ~0**(직교성 문서 프론티어 법칙). 기록용 보관.
