# 📂 세션1 공유 번들 — INDEX (KML 2026S 성별예측)

> 세션1(minhyung)의 mega 피처 + 모든 시도본 + 분석 문서 통합. 2026-06-06.
> **한 줄 결론: 단일 천장 ~0.72 OOF / ~0.73 LB 최종확정. FE·딥·외부·아키텍처 전부 강한 base에서 음성. 본진=GBT/AG, 전략=consolidate.**

## 폴더 구성
```
세션1_공유_mega피처/
├── INDEX.md                         ← 이 파일
├── mega_train/test.parquet          ← mega 572피처 행렬 (custid)
├── mega_columns_manifest.csv        ← TE컬럼 표시 (f219-238, f545-560)
├── mega_folds.csv                   ← seed42 5fold (TE 안전용)
├── README_mega피처_사용법.md
├── feature_attempts/                ← 피처 시도본 전부
│   ├── block_s1_{interact,cross,brandext,ext,text}_{tr,te}.parquet
│   ├── model_oof_attempts/          ← 모델 OOF 멤버 14종 (증강딥·랭킹·텍스트·transduct)
│   └── README_피처시도본.md          ← 강한-base Δ + AUC 매니페스트
└── docs/                            ← 분석 문서 8종 (아래)
```

## docs/ 핵심 문서
| 문서 | 내용 |
|---|---|
| **세션1_단일모델_탐색결론.md** | "FE는 강한 base서 검증" 교훈 + 전 방향 음성 종합 |
| **07_직교성_실험_종합.md** | ⭐프론티어 법칙 `AUC≈0.55+0.18·corr` = 강+직교 멤버 부재 증명 |
| **세션1_셰이크다운_분석.md** | 1등 갭(+0.0024) < public낙관(+0.0026)≈노이즈 → public추격 위험 |
| **06_과거대회_토론_종합정리.md** | 8년 5대회 0.73 천장 / 0.81은 다른데이터·수업통합앙상블 |
| **FE_가설_정리.md** | 58 FE블록 출발가설 전체 |
| 세션1_딥증강_README.md | 증강 9종: adv 0.7148(GBT 넘김)·앙상블 기여 0 |
| session1_FE_README.md | cross/brandext/ext/text 블록 상세 |
| 룰확인_필요.md | 외부지식 룰 게이트 질문 |

## 검증된 핵심 결론 (전부 실측)
1. **단일 천장 ~0.72 OOF / ~0.73 LB** — 8년 대회 + 1등코드 0.714 + 모든 부분집단 ≤0.734.
2. **강+직교 멤버 부재** (프론티어 법칙, NCL·잔차로 증명) → 앙상블 천장 0.723도 못 뚫음.
3. **FE 전부 강한 base 음성**: cross/brandext/ext/text −0.001~−0.0017. **interact만 +0.00075(⚠️선택과적합 미검증).**
4. **딥**: 증강(adv) 0.7148로 단일 GBT 넘으나 redundant(corr0.955)→앙상블 기여 0. 시퀀스 0.66 캡.
5. **외부지식**: GBT엔 −(자동TE중복), 약한딥엔 +0.0035지만 약해서 무기여.
6. **셰이크 리스크**: 1등 추월은 노이즈 영역 → robust 헤지(메인 42_nestedCaruana + 헤지 24_hillclimb).

## 권장 최종 전략
**Consolidate.** 추가 실험 무익(근거상). Kaggle 재인증 → 2픽 락(메인=nested-Caruana robust, 헤지=hillclimb 최고LB). 미검증 1건=interact nested-CV.

---
*세션1. 재현 스크립트: `s2_공유/session1_FE/scripts/`. 멤버 OOF: `feature_attempts/model_oof_attempts/`.*
