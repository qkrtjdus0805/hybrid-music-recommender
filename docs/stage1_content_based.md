# 1단계 Content-Based Filtering — 실행 안내


## 구현 범위

- **1-1 StandardScaler**: 12개 Spotify 음향 특성의 Z-score 표준화
- **1-2 TF-IDF**: `genre`, `tags`를 텍스트 벡터로 변환 후 음향 특성과 결합 (`tag_weight=0.4`)
- **1-3 K-Means++**: 음향 특성으로 음악을 6개 군집으로 분류하고 inertia 및 표본 silhouette 측정
- **1-4 사용자 프로필 + Cosine Similarity**: 학습 청취 기록에서 `log1p(playcount)` 가중 평균 취향 벡터 구성 후 `S_CB(user, track)` ∈ [0,1] 반환; 이미 청취한 곡은 추천에서 제외

## 준비

Python 3.10+ 권장. `pip install -r requirements.txt`

## 실행 (실제 데이터, 빠른 시연)

```bash
python content_based.py --music "data/Music Info.csv" --history "data/User Listening History.csv.zip" --user-fraction 0.005
```

전체 971만 청취 기록을 읽어 **약 0.5%의 사용자를 일관되게 추출**합니다. 같은 사용자의 청취는 묶어서 유지하고, 사용자별 노래의 80%를 학습, 20%를 테스트용으로 분리합니다. **개인화 프로필은 TRAIN으로만 학습합니다.** 나머지 99.5% 사용자 정보는 모델에 사용하지 않습니다. (50,683곡의 음향 특징으로 K-Means는 전부 학습합니다.)

실행하면 `outputs/content_based/`에 다음을 생성합니다:

- `clustering_metrics.json` : 군집 크기, inertia, 표본 silhouette
- `song_clusters.csv` : 각 노래의 군집 번호
- `demo_train_history.csv` / `demo_test_history.csv` : 테스트용 분할 청취 기록
- `personalized_top10.csv` : 실제 사용자에게 추천된 상위 10곡

K를 비교해 군집 수 선정 근거를 남기고 싶다면 다음 명령에서 `--compare-k`를 추가하세요. `kmeans_k_comparison.csv`에 K=4,6,8,10의 inertia와 표본 silhouette가 저장됩니다.

## 팀원과 공통 분할을 사용할 때

**각 담당자가 서로 다른 80/20 분할을 만들어서는 공정하게 비교할 수 없습니다.** 팀에서 합의한 `shared_train_history.csv`, `shared_test_history.csv`를 공통으로 생성해주세요. 파일이 확정되면 아래처럼 1번 모델에 학습 이력을 전달하면 됩니다.

```bash
python content_based.py --music "data/Music Info.csv" --train-history "data/shared_train_history.csv"
```

`--train-history`는 테스트 파일을 건드리지 않습니다. 2번 MF 모델과 3번 평가 담당자도 같은 train/test 파일을 사용해야 합니다.

## 3번 Hybrid 담당자에게 제공할 API

```python
from content_based import ContentBasedRecommender
import pandas as pd

music = pd.read_csv("data/Music Info.csv")
train = pd.read_csv("data/shared_train_history.csv")
cbf = ContentBasedRecommender(n_clusters=8).fit(music, train)

# 같은 행 순서의 후보 음악을 평가, 결과는 [0, 1] 추천 점수 배열
s_cb = cbf.score_user_items(user_id, candidate_features=cbf.track_features)

# 실제 추천
print(cbf.recommend_user(user_id, top_n=10))

# 새로운 음악 외부 데이터: music 스키마에 맞게 이름/장르 컬럼을 변환한 뒤
# 동일한 학습 스케일러와 TF-IDF를 이용해 변환
features, clusters = cbf.transform_tracks(new_songs_df)
s_cb_cold = cbf.score_user_items(user_id, candidate_features=features)
```

**주의:** 사용자 `user_id`는 문자열로 통일하세요. 협업 점수 `S_MF`와 `S_CB`의 행 순서를 반드시 맞춰야 합니다.

