# 📦 세션1 FE 공유 (→ 세션2)

> custid 정렬 = train/test `custid.unique()` (세션2 블록·mega와 동일, fold5/seed42 호환).
> 전부 fold-safe(누수0). base/mega/megaA/s2_pool_sel 위에 그대로 concat 가능.
> 작성: 세션1, 2026-06-05.

## 1) `blocks/block_s1_cross_{tr,te}.parquet` — ⭐ 취향 부호역전(선물) 6피처
**핵심 통찰**: 라벨=카드주인 성별. 남성이 여성복을 선물로 사면 취향(여성)↔라벨(남성) 역전.
→ 습관과 반대방향(교차) 구매를 격리하면 취향의 **반대 정보**가 된다.

| 검증 | 결과 |
|---|---|
| `cross_ratio`(습관 반대방향 구매비중) 단독 | AUC **0.42 (강한 역전, flip 0.58)** |
| 트리 증분 (taste 단독 위) | +0.0033 |
| **mega(572) 위 순증분** | **+0.00111** (conflict/dispersion/det 있는데도) = 새 신호 |

비교: 세션2 잔존 최고블록 pricecat +0.0007 / taxonomy +0.0008 / supseg +0.0010 → **동급~상위.**

**피처(6)**: cross_ratio, cross_taste, sharp_share, sharp_taste, sharpS_share, taste_H.
- corner 성별rate: train=fold-safe(5fold seed42, α=20), test=full-train TE. habit=2회+&비선물시즌, cross=(rate−base)×habit방향<0, sharp=cross&고가&일회성. 선물시즌=발렌타인/화이트/가정의달/연말.
- 재현: `scripts/build_cross_block.py`. 진단근거: `phase0b/c/d.py`.
- ⚠️ **다레벨 확장 권장**(brd/goodcd): 고카디일수록 "여성복 선물" 교차가 선명 → 더 큰 게인 기대. (현 corner-only)

## 1b) `blocks/block_s1_ext_{tr,te}.parquet` — ⚠️외부 상품-타깃-성별 11피처 (룰 게이트)
**가설**: 내부 성별률은 선물로 base쪽 압축됨(수입화장품 0.32·골프 0.46). **외부 세계지식의 "상품 타깃성별"(1=남성)** 은 비압축 → 그 차이가 선물신호.
- 피처: pc/corner별 외부타깃(amtw·cnt·강여성/강남성 비중·std) + deconf_gap(외부−내부율). 라벨=Claude 세계지식(pc 72·corner 59개).
- **mega 위 +0.00118** (cross와 동급). 단 ⚠️ **외부지식 = 룰 회색(호스트 확인 필수)**. 재현: `scripts/proto_external_v2.py`, `build_external_block.py`.
- 🔴 **cross와 stack 안 됨**: `+cross+external = −0.00014`(둘 다 같은 "선물 갭" 신호 → 중복). **cross 또는 external 중 하나만** 쓸 것.

## 1c) `blocks/block_s1_brandext_{tr,te}.parquet` — ⭐⭐외부 브랜드-타깃-성별 5피처 (최고 외부레버, 룰 게이트)
**가설**: 브랜드는 고카디(1882)=내부 TE 과적합 롱테일. 외부 브랜드지식이 거기서 가치.
- **mega 위 +0.00231** (cross·corner_ext의 2배, 역대 최고 외부 증분). 라벨=Claude 세계지식 **174개 인식브랜드**(비비안=란제리, 잭니클라우스=골프 등).
- 🔴 **핵심 교훈**: 가치는 **커버리지가 아니라 브랜드-특정 지식.** 모르는 브랜드를 **주력 카테고리로 추론해 전수커버(1882) 하면 −0.0002로 죽음**(코너 재탕=내부TE 중복, 브랜드신호 희석). → **LLM은 "진짜 아는 브랜드만" 라벨, 모르는 건 미라벨로 비울 것.**
- 재현: `scripts/proto_brand_external.py`(증분), `proto_brand_full.py`(전수커버 실패 증거). 헤드룸: LLM이 174개 너머 인식브랜드 더 라벨하면 더 오를 여지.
- ⚠️ 룰 게이트(외부지식). cross와 부분중복(같은 "라벨=카드주인" 뿌리).

## 2) `blocks/block_s1_text_{tr,te}.parquet` — 텍스트 직교멤버 48피처
고객 "구매 내러티브 문서"(코너+브랜드+파트명, 지출/빈도 top) → 한국어 SBERT(jhgan/ko-sroberta-multitask, MPS frozen) 768 → SVD48.
- 단독 xgb OOF **0.65118**, corr(best 0.722) **0.710**. (비지도, RunPod 불필요)
- oof/test: `oof_members/{oof,test}_sbert_svd48.npy`. 재현: `scripts/build_text_member.py` (임베딩 캐시 `text_emb/`).
- 용도: 약하나 **텍스트축 다양성**(앙상블 직교멤버). 강화여지: 한국어 도메인 LM 파인튜닝 / 더 긴 문서.

## 3) `scripts/phase0*.py` — 천장 진단 (참고)
단일 천장 ~0.72~0.73 근거: 모든 부분집단 AUC≤0.734, 정직취향TE결합 0.648, **leaky goodcd +0.052=전부 누수**(이두영 케이스 수치확정). 비지도 임베딩/그래프 ~0. → **지도·수작업 방향만 유효**(cross가 그 산물).

---
*문의/분담: 세션1. cross 다레벨(brd/goodcd)은 세션2와 분담 환영.*
