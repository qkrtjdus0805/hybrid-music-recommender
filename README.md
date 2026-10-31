# hybrid-music-recommender
2026 team 11 Machine Learning Term Project · Fall 2026

## Objective

음악의 음향 특징과 사용자 청취 이력을 결합해 Cold-Start 문제를 개선하는 하이브리드 음악 추천 시스템

## Methods

- **Stage 1: Content-Based Filtering** — StandardScaler, TF-IDF, K-Means++, Cosine Similarity
- **Stage 2: Collaborative Filtering** — Sparse Log Playcount, Matrix Factorization (Implicit ALS by default, Biased SVD via SGD optional), Per-user Min-Max Normalization
- **Stage 3: Weighted Hybrid Filtering** — Adaptive Weighting $\alpha(i)$ (Hyperbolic decay based on listener count), Top-N Recommendation
- **Datasets**: MSD + Spotify + Last.fm (`Music Info.csv`, `User Listening History.csv.zip`)
- **Evaluation**: Silhouette Score, RMSE/MAE, Precision@K, Recall@K, NDCG@K, HitRate@K

---

## Project Structure

```
├── Music Info.csv                  # 50,683곡 음향 특성 및 메타데이터 카탈로그
├── README.md                       # 프로젝트 개요 및 전체 안내
├── requirements_content_based.txt  # 1단계 요구 패키지
├── requirements_collaborative.txt  # 2단계 요구 패키지
│
├── content_based.py               # 1단계 콘텐츠 기반 필터링 모델 및 학습 코드
├── test_content_based.py          # 1단계 단위 테스트 코드
├── README_content_based.md        # 1단계 상세 가이드 및 API 명세
│
├── collaborative.py               # 2단계 협업 필터링 모델, 학습, 평가 코드
├── test_collaborative.py          # 2단계 간단한 단위 테스트
└── README_collaborative.md        # 2단계 상세 가이드 및 API 명세
```

---

## Quick Start

### 1단계: Content-Based Filtering
- 상세 안내: [README_content_based.md](file:///home/ho/coding/python/hw/ml/hybrid-music-recommender/README_content_based.md)
- 학습 및 추천:
  ```bash
  python content_based.py --music "Music Info.csv" --history "User Listening History.csv.zip" --user-fraction 0.005
  ```
- 단위 테스트:
  ```bash
  python -m unittest test_content_based.py -v
  ```

### 2단계: Collaborative Filtering
- 상세 안내: [README_collaborative.md](file:///home/ho/coding/python/hw/ml/hybrid-music-recommender/README_collaborative.md)
- 학습 및 추천:
  ```bash
  python collaborative.py --music "Music Info.csv" --history "User Listening History.csv.zip" --user-fraction 0.005
  ```
- 단위 테스트:
  ```bash
  python -m unittest test_collaborative.py -v
  ```
- 공통 분할로 학습 + 평가 (Precision/Recall/NDCG@K):
  ```bash
  python collaborative.py --train-history "shared_train_history.csv" --test-history "shared_test_history.csv"
  ```
