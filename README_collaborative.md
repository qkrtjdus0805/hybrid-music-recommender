# 2단계 Collaborative Filtering — 실행 안내

`collaborative.py`(모델/학습/평가)와 `test_collaborative.py`(간단한 단위 테스트)로 구성됩니다. 데이터 로딩과 train/test 분할은 1단계 `content_based.py`의 함수를 그대로 사용하므로 두 모델의 분할이 동일합니다.

## 구현 범위

- **2-1** `r = log(1 + playcount)` 변환
- **2-2** 행렬 분해: **Implicit ALS (기본값)** 또는 Biased SVD (`--method sgd`, numba가 있으면 자동 가속)
- **2-3** 사용자별 Min-Max 정규화로 `S_MF(user, track)` ∈ [0,1]; 학습에 없던 곡(Cold-Start)은 0.0
- **3-2** 평가: Precision@K, Recall@K, NDCG@K (sgd는 RMSE/MAE 추가). 이미 학습에서 들은 곡은 제외

## 준비

`pip install -r requirements_collaborative.txt` (`numba`는 선택, SGD 속도 향상용)

## 실행

```bash
# 데모: 원본 이력에서 0.5% 사용자 추출 + 80/20 분할 후 학습/평가
python collaborative.py --music "Music Info.csv" --history "User Listening History.csv.zip"

# 팀 공통 분할 사용 (권장)
python collaborative.py --music "Music Info.csv" --train-history shared_train_history.csv --test-history shared_test_history.csv

# SGD(Biased SVD)로 실행
python collaborative.py --music "Music Info.csv" --history "User Listening History.csv.zip" --method sgd
```

주요 옵션: `--method {als,sgd}`(기본 als), `--factors 32`, `--epochs 20`, `--reg 0.05`, `--lr 0.015`(sgd), `--user`, `--top-n`.

`collaborative_output/`에 `cf_evaluation_metrics.json`(테스트 이력 지정 시), `personalized_top10_cf.csv`가 생성됩니다.

## 테스트

```bash
python -m unittest test_collaborative.py -v
```

## 3번 Hybrid 담당자에게 제공할 API

```python
from collaborative import CollaborativeRecommender
import pandas as pd

music = pd.read_csv("Music Info.csv", dtype={"track_id": str})
train = pd.read_csv("shared_train_history.csv")
cf = CollaborativeRecommender().fit(train, music)        # method="als"가 기본

ids = music["track_id"].tolist()
s_mf = cf.score_user_items(user_id, ids)                 # [0,1], 미학습 곡은 0.0
alpha = cf.compute_dynamic_alpha(ids)                    # alpha(i) = 0.25 + 0.75 / (1 + n_i/25)
s_hybrid = alpha * s_cb + (1 - alpha) * s_mf
print(cf.recommend_user(user_id, top_n=10))
```

**주의:** `user_id`는 문자열로 통일하고, `S_CB`와 `S_MF`의 후보 곡 순서를 반드시 맞추세요. 테스트 이력은 학습에 사용하면 안 됩니다.
