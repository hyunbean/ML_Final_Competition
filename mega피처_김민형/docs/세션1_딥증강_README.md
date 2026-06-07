# 🧪 세션1 딥러닝 증강 실험 (2026-06-06)

> 목적: 데이터 증강·자기지도로 딥(MLP)을 GBT(0.7137)에 근접/추월시킬 수 있나. (오버피팅 감안)
> base = 강한 MLP on s2_pool_sel(360). 전부 fold5/seed42 honest OOF, MPS.
> npy: `npy/oof_mlp_<method>.npy` + `test_mlp_<method>.npy` (앙상블 멤버 후보)

## 기준선
| | OOF AUC |
|---|--:|
| **GBT (s2_pool_sel xgb)** | **0.7137** ← 목표 |
| MLP base (증강 없음) | 0.7088 |

## 1차 증강 (proto_deep_augment.py)
| 방법 | 설명 | OOF |
|---|---|--:|
| mixup | 입력+라벨 선형보간(Beta) 정규화 | 0.7096 |
| noise | 입력 가우시안노이즈 0.1 | 0.7109 |
| **ssl(mask)** | train+test 마스크30%-복원 사전학습→finetune | **0.7110** |
| ssl+mixup | 결합 | 0.7109 |

## 2차 증강 (proto_deep_augment2.py) — 다양화
| 방법 | 설명 | OOF |
|---|---|--:|
| **adv** ⭐ | FGSM 적대적 증강(perturbation robust) | **0.71481 (GBT 넘음 +0.0011)** |
| **manifold** ⭐ | 히든층 mixup(은닉 표현 보간) | **0.71432 (GBT 넘음 +0.0006)** |
| swapdae | Porto Seguro: 컬럼별 값 스왑 손상→DAE 복원 사전학습 | 0.71360 ≈GBT |
| vime | 마스크-복원 + 마스크-예측 멀티태스크 SSL | 0.71350 ≈GBT |
| cutmix | 피처 부분집합을 다른 샘플과 스왑(라벨 비율혼합) | 0.71278 |
| smote | 소수클래스 보간 오버샘플 | 0.70625 (해로움) |

## 핵심 결론 (2차 갱신 — 반전)
- **강한 증강은 딥을 GBT 위로 올린다**: base 0.7088 → **adv 0.71481 / manifold 0.71432 > 단일 XGBoost 0.7137.** 적대적 학습·manifold mixup이 딥의 데이터-기근을 강하게 보상 → **처음으로 딥이 단일 GBT를 넘음.**
- **단, AutoGluon(0.722)은 못 넘음** — 최강 단일은 여전히 AG. 딥 0.715 < AG 0.722.
- **천장(~0.72 OOF)은 그대로** — 증강은 정규화일 뿐 새 신호 생성 아님. adv도 0.715에서 멈춤.
- **SMOTE 오버샘플은 해로움**(0.706) — 합성 소수클래스가 노이즈 주입.

## 앙상블 검증 (corr·Caruana 기여)
- 증강딥 corr(GBT best) = **0.95~0.96** (같은 피처 위 딥 → 직교 아님).
- Caruana 기존풀 0.72152 → +증강딥 **0.72169 (Δ+0.00017, 노이즈)**.
- 단 **Caruana가 증강딥을 선택**(adv 0.06·manifold 0.058·swapdae 0.018·vime 0.022 = ~16%) → 약한 딥(0.66, 가중 0)과 달리 **앙상블에 낄 만큼 강해짐**. 그러나 redundant라 게인은 미미.

## 최종 결론
**딥 ≠ 이 데이터와 안 맞음 — 강한 증강(adv FGSM·manifold mixup)으로 딥을 단일 GBT 위(0.7148 > 0.7137)로 올렸고, Caruana 선택까지 됨.** 그러나:
- ① AG 단일(0.722) 못 넘음 ② 앙상블 천장 0.723 못 넘음 ③ corr 0.95라 직교기여 +0.0002(노이즈).
- **"강함+직교" 스위트스팟 부재**: 직교 딥(시퀀스 0.66)은 약하고 강한 딥(0.715)은 redundant. 천장 ~0.72는 모델·증강 무관.
- 미시도 1수: **강한 증강 × 직교 아키텍처(set-transformer/entity-NN)** = 강함+직교 동시 노림. 단 시퀀스 4번 0.66캡 근거상 기대 낮음.

RunPod 불필요(전부 MPS 수분). 최고 증강 = **adv(FGSM) 0.71481**.

---
*재현: `~/Desktop/dacon final competition/proto_deep_augment{,2}.py`. 2차 수치는 `results.json` 참조.*
