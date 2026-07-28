# 포스트모템: 피처 누수(leakage) 발견 + Public-Private 역전(public-overfit)

> 이 문서는 `POSTMORTEM_TEMPLATE.md`를 이 레포의 기존 회고 문서(`docs/retrospective.md` 섹션 4·17~19, `docs/shakeup-analysis.md`)에 이미 서술되어 있던 두 건의 실제 사건에 맞춰 채운 것입니다. 새로운 사실을 추가하지 않았습니다.

이 레포에는 성격이 다른 두 사건이 함께 기록되어 있어 하나의 문서에 두 절로 나눠 정리합니다: (A) 피처 계산 시 발생한 데이터 누수, (B) 대회 최종 결과에서의 Public 1위 → Private 2위 역전(public-overfit).

---

## A. 피처 누수(leakage) 발견 — `lb_g6_skew`

### 발생 현상 (What Happened)
신규 피처 `lb_g6_skew`가 검증 과정에서 성능을 +0.005 끌어올리는 것으로 관측되었으나, 이 값이 fold 분리 없이 **전체 train 데이터로 계산**되어 있었음이 이후 확인되었습니다. 출처: `docs/retrospective.md` 섹션 4 "검증으로 가짜 걸러냄" — "신규 피처 lb_g6_skew가 전체 train으로 계산돼 +0.005를 가짜로 부풀린 것 적발".

### 발견 경위 (How It Was Found)
회고 문서에는 "leak 점검" 과정에서 이 피처가 적발되었다고만 기록되어 있고, 구체적으로 어떤 절차·누가 발견했는지에 대한 상세 서술은 **이 저장소의 기존 문서에서 확인되지 않음**.

### 원인 분석 (Root Cause)
피처가 전체 train 데이터를 대상으로 한 번에 계산되어, 각 fold의 검증(validation) 구간 정보가 해당 fold의 학습(train) 피처 계산에 스며든 전형적인 target/aggregate leakage 패턴입니다. 근거: `docs/retrospective.md` 섹션 4의 "전체 train으로 계산돼 ... 가짜로 부풀린 것 적발" 서술.

### 조치 (Remediation)
KFold 방식의 OOF(out-of-fold) 계산으로 해당 피처를 재계산했습니다. 그 결과 최초 관측되었던 +0.005 이득이 실제로는 +0.00156로 축소되었습니다. 출처: `docs/retrospective.md` 섹션 4 — "KFold OOF로 수정하니 +0.00156로 축소."

### 재발 방지책 (Prevention)
회고 섹션 6·7·11(핵심 교훈)에서 "검증이 발견만큼 중요", "누수 검증(leak/adversarial/fold 정렬)은 항상"을 반복 명시하고 있습니다(섹션 17 결론부 8번 항목: "누수 검증(leak/adversarial/fold 정렬)은 항상"). 다만 이를 강제하는 자동화된 테스트/스크립트가 이 레포에 존재하는지는 **이 저장소의 기존 문서에서 확인되지 않음** — 문서화된 원칙으로만 남아 있고, 코드 레벨의 강제 장치는 미적용으로 보입니다.

### 교훈 (Lessons Learned)
새로 만든 피처가 만들어낸 성능 향상은 fold 분리를 지키지 않은 계산 방식의 아티팩트일 수 있습니다. "검증이 발견만큼 중요하다"는 이 프로젝트를 넘어 일반화 가능한 교훈입니다(`docs/retrospective.md` 섹션 7 6번째 항목).

---

## B. Public 1위 → Private 2위 역전 (public-overfit)

### 발생 현상 (What Happened)
최종 제출 기준 Public 리더보드 0.7385(10팀 중 1위)였으나, Private 리더보드에서는 0.7324로 2위로 밀려났습니다(1위 팀 Private 0.7345). 출처: `README.md` "📊 결과" 표, `docs/retrospective.md` 섹션 17 표.

### 발견 경위 (How It Was Found)
대회 종료 후 Private 리더보드가 공개되며 순위 역전이 확인되었고, 이후 `docs/retrospective.md` 섹션 17~19와 `docs/shakeup-analysis.md`에서 1위·3위 팀의 공개 발표자료와 자체 제출 이력(140회)을 비교 분석했습니다.

### 원인 분석 (Root Cause)
- **갭–public 상관 차이**: 자체 팀은 갭(Private−Public)과 Public 점수의 상관이 +0.63으로, Public이 오를수록 갭도 커지는 전형적 public-overfit 패턴이었던 반면, 1위 팀은 −0.19로 오히려 안정적(robust)이었습니다. 같은 Public 0.737~0.738대에서 갭 차이 약 0.003이 순위를 뒤집었습니다. 출처: `docs/retrospective.md` 섹션 17.
- **과다한 제출 횟수**: 자체 팀 140회 제출 vs. Private 3위 팀 47회 제출. Public LB를 검증셋처럼 사용하며 튜닝할수록 갭이 커졌습니다(+0.0043 → +0.0071 수준으로 확대). 출처: 섹션 17.
- **Pseudo-labeling 의존**: pseudo-labeling은 test-side 신호를 학습에 사용하므로 Public에 fit되기 쉬우며, 1위·3위 팀은 모두 pseudo-labeling을 사용하지 않았습니다. 출처: 섹션 17, 섹션 7.
- **Drift/Gap 관리 부재**: 1위 팀은 feature 선정 시 train/test 분포차(drift)를 제한(DRIFT_MAX=0.35)하고 균등 rank를 25% 섞어 public 과적합을 능동적으로 억제했으나, 자체 팀은 CV−LB 갭을 제출 시 표로 추적하지 않았습니다. 출처: 섹션 17.

### 조치 (Remediation)
대회 종료 후 조치이므로 실시간 교정은 이루어지지 않았습니다. 사후적으로 1위·3위 팀의 발표자료를 확보해 정량 비교 분석 문서(`docs/shakeup-analysis.md`, `docs/retrospective.md` 섹션 17~19)를 작성해 원인을 구조화했습니다.

### 재발 방지책 (Prevention)
`docs/retrospective.md` 섹션 6·11·19에 다음 항목이 다음 대회를 위한 재발 방지책으로 명시되어 있습니다: (1) 모든 제출에 CV·LB·Gap을 표로 기록하고 Gap이 벌어지면 해당 제출은 채택하지 않을 것, (2) drift-aware feature selection 적용, (3) 균등 rank를 섞어 public 과적합을 능동 억제, (4) pseudo-labeling은 신중히 사용하거나 회피, (5) adversarial validation을 대회 초반에 먼저 수행해 train/test 분포차를 확인할 것(섹션 19). 이 항목들이 다음 대회에서 실제로 적용되었는지는 이 레포의 범위를 벗어나며 **이 저장소의 기존 문서에서 확인되지 않음**(다음 대회 코드는 이 레포에 없음).

### 교훈 (Lessons Learned)
- "Public 1위 ≠ 승리." 같은 Public 점수대에서도 갭–public 상관 구조가 다르면 Private 순위가 뒤집힐 수 있습니다(`docs/retrospective.md` 섹션 7 1번째 항목).
- "제출 수는 과적합 지표다." Public LB를 검증셋처럼 반복 사용할수록 갭이 커집니다(섹션 7 2번째 항목). 이 교훈은 이 프로젝트를 넘어, 반복적으로 held-out 신호에 노출되는 모든 모델 선택/튜닝 프로세스에 일반화 가능합니다.

---

*원본 서술 위치: `docs/retrospective.md` 섹션 4(피처 누수), 섹션 17~19(public-overfit 사후분석), `docs/shakeup-analysis.md`(정량 시각 분석), `README.md`(결과 요약), `MODEL_CARD.md`(성능/한계 절).*
