신현빈 제출 블렌드 (stack3) oof+test npy — 김민형 Caruana용
custid 순서: custid_train.npy(30000), custid_test.npy(19995)

각 제출별 _oof.npy(train OOF, 30000) + _test.npy(test, 19995):
  49_submission_stack3_hyunbin  LB 0.73080  [정확 재현, rank-corr 1.000]
  47_stack3_txnfix_hyunbin      LB 0.73021  [정확 재현, rank-corr 1.000]
  43_stack3_tfidf_hyunbin       LB ~0.730   [근사, rank-corr 0.9995 — txn누수수정 전이라 정확재현 불가, 사실상 동일]

※ _oof와 _test는 같은 블렌드 레시피의 짝이라 그대로 Caruana 멤버로 사용 가능.
※ 추가로 우리 단일모델 OOF는 repo의 artifacts/oof/ 에 자동저장됨(first_cat, first_lgbm 등) → git pull로 사용.
