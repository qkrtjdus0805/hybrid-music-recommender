# Hybrid Music Recommender
**2026 Team 11 Machine Learning Term Project · Fall 2026**

음악의 **음향·태그 특징(Content-Based)**과 **사용자 청취 이력 및 아티스트 신호(Collaborative + Artist Affinity)**를 결합하여, 협업 필터링의 **Cold-Start 문제**와 **인기곡 편중(Popularity Bias)**을 개선한 하이브리드 음악 추천 시스템입니다.

---

## 주요 결과 요약

| 평가 데이터셋 및 지표 | 단독 CB | 단독 MF (ALS) | 제안서 공식 (동적 $\alpha$) | **최종 모델 (`Final`)** |
| --- | :---: | :---: | :---: | :---: |
| **MSD 내부 평가 — `NDCG@10`** | 0.0011 | **0.1814** | 0.0158 | **0.1210** |
| **MSD 내부 평가 — 모의 신곡 `Recall@20`** | 0.0000 | 0.0000 | 0.0000 | **0.0632** |
| **MSD 내부 평가 — 비주류(1~4명) `Recall@20`** | 0.0005 | 0.0138 | 0.0041 | **0.1488** |
| **MSD 내부 평가 — Top-10 인기곡(25명+) 비율** | 7.9% | 92.6% | 2.3% | **31.2%** |
| **Spotify 외부 80/20 검증 — `Hit Rate@10`** | 11.3% | 1.7% | 11.1% | **55.0%** |
| **Spotify 외부 80/20 검증 — `NDCG@10`** | 0.0164 | 0.0024 | 0.0161 | **0.1749** |

---

## 문서 안내 (`docs/`)

보고서 작성이나 코드 파악 전 **[전체 프로젝트 총괄 문서 (`docs/project_overview.md`)](docs/project_overview.md)**를 먼저 읽으면 전체 흐름과 수식, 실험 과정을 한눈에 파악할 수 있습니다.

- **[`docs/project_overview.md`](docs/project_overview.md)** — **전체 프로젝트 상세 기술·실험·결과 분석 총괄 문서 (보고서 레퍼런스용)**
- [`docs/stage1_content_based.md`](docs/stage1_content_based.md) — 1단계 Content-Based Filtering (StandardScaler, TF-IDF, K-Means++)
- [`docs/stage2_collaborative.md`](docs/stage2_collaborative.md) — 2단계 Collaborative Filtering (Implicit ALS / Biased SVD, 공출현 결합)
- [`docs/stage3_hybrid.md`](docs/stage3_hybrid.md) — 3단계 Hybrid 개선 실험(실험 1~5) 및 내부 심층 검증(검증 1~3) 상세 수치
- [`docs/val3_spotify_80_20.md`](docs/val3_spotify_80_20.md) — 외부 데이터셋 검증: Spotify 플레이리스트 80/20 이어듣기 평가
- [`docs/case_study.md`](docs/case_study.md) — 사례 연구: 12차원 음향 공간에서의 우리 최종 모델 vs. Spotify vs. Tidal 비교

---

## 저장소 구조

```text
├── content_based.py           # 1단계: 음향(12차원) + 장르/태그 TF-IDF 기반 콘텐츠 추천
├── collaborative.py           # 2단계: Implicit ALS / Biased SVD + 지역 아이템 공출현 협업 추천
├── hybrid.py                  # 3단계 기준선: 제안서 동적 α 결합, 모의 신곡 평가기, 실험 1
├── final_model.py             # 3단계 최종 모델: ArtistCB(β=0.9) + ArtistMF(γ=5) + 아티스트 상한(cap=5)
├── make_figures.py            # 실험 결과 시각화 생성기
├── data/                      # 원본 데이터셋(Music Info.csv, User Listening History, spotify_songs) 및 사례 연구 입력
├── docs/                      # 전체 프로젝트 총괄 해설서 및 단계별·검증별 보고 문서
├── experiments/               # 가설 검증 실험(exp2~exp5), 심층·외부 검증(val1~val3), 상용 비교(case_study)
├── scripts/                   # 보조 스크립트 (사례 연구 음향 특징 추출 등)
├── tests/                     # 핵심 모듈 단위 테스트
└── outputs/                   # 실험 평가 지표(JSON/CSV) 및 시각화 결과물(PNG)
```

---

## 빠른 시작 (Quick Start)

모든 명령어는 **저장소 최상위 경로(루트)**에서 실행합니다.

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 전체 단위 테스트 실행 (데이터셋 없이 합성 픽스처로 170개 테스트 즉시 검증)
python -m unittest discover -p "test_*.py"

# 3. 공통 학습/테스트 분할 생성 (data/ 폴더에 원본 데이터 배치 후 실행)
python hybrid.py make-split --music "data/Music Info.csv" --history "data/User Listening History.csv.zip" --user-fraction 0.05

# 4. 최종 모델 평가 및 사용자 맞춤 Top-10 추천
python final_model.py evaluate --music "data/Music Info.csv" --train-history data/shared_train_history.csv --test-history data/shared_test_history.csv
python final_model.py recommend --music "data/Music Info.csv" --train-history data/shared_train_history.csv --user USER_ID --top-n 10

# 5. 외부 Spotify 플레이리스트 80/20 검증 및 상용 서비스(Spotify/Tidal) 비교 사례 연구
python -m experiments.val3_spotify_80_20.spotify_playlist_experiment
python scripts/build_case_study_features.py
python -m experiments.case_study.case_study_experiment
```
