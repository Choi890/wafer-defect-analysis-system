# Wafer Defect Analysis System

한국어명: 반도체 Wafer 불량 분석 및 품질 모니터링 시스템

Wafer Map 데이터를 전처리하고 CNN 기반 분류 모델로 불량 패턴을 예측한 뒤, 결과를 SQLite DB에 저장하고 FastAPI와 Streamlit 대시보드로 제공하는 end-to-end 제조 품질 분석 시스템입니다.

## 1. 프로젝트 개요

이 프로젝트는 단순 모델 학습 노트북이 아니라 제조 데이터를 처리하는 IT 시스템 흐름을 보여주는 포트폴리오용 프로젝트입니다.

- Wafer Map 이미지형 데이터 생성/로드
- 전처리, 라벨 인코딩, train/test split
- PyTorch CNN 모델 학습 및 성능 평가
- 예측 결과, 모델 지표, Lot별 통계를 SQLite에 저장
- FastAPI API 제공
- Streamlit 대시보드와 Excel/PDF 리포트 제공

실제 `data/raw/wafer_map_dataset.pkl` 파일이 없으면 데모용 synthetic wafer map 데이터가 자동 생성됩니다.

## 2. 개발 배경

반도체 품질 데이터는 Lot, Wafer, Die 단위로 추적되고, 불량 위치의 공간적 패턴은 공정 이상을 파악하는 중요한 단서가 됩니다. 이 프로젝트는 Wafer Map을 이미지처럼 처리해 불량 유형을 분류하고, 분석 결과를 조회/시각화/리포트화하는 흐름을 구현합니다.

## 3. 시스템 구조

```text
Raw Wafer Data
  -> Preprocessing
  -> CNN Model Training / Prediction
  -> SQLite Persistence
  -> FastAPI Backend
  -> Streamlit Dashboard
  -> Excel/PDF Reports
```

주요 폴더:

```text
data/              raw, processed, sample 데이터
src/data/          데이터 로드 및 전처리
src/models/        CNN 모델, 학습, 평가, 예측
src/database/      SQLite 스키마 및 repository
src/api/           FastAPI 서버
src/dashboard/     Streamlit 대시보드
src/utils/         시각화, metric, report 생성
docs/              아키텍처, ERD, API 문서
reports/           생성된 Excel/PDF/metric 결과
saved_models/      학습된 PyTorch 모델
```

## 4. 데이터 흐름

입력 데이터 컬럼:

| 컬럼 | 설명 |
| --- | --- |
| `wafer_id` | Wafer 식별자 |
| `lot_id` | Lot 식별자 |
| `wafer_map` | 2D wafer map 배열. 0=outside, 1=normal die, 2=defect die |
| `failure_type` | 실제 불량 유형 |
| `die_size` | wafer 내 die 개수 |
| `inspection_date` | 검사 일자 |

분류 대상:

```text
Normal, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch
```

## 5. 모델 구조

PyTorch CNN 구조:

```text
Input 1x32x32
  -> Conv2D + BatchNorm + ReLU + MaxPool
  -> Conv2D + BatchNorm + ReLU + MaxPool
  -> Conv2D + BatchNorm + ReLU + AdaptiveAvgPool
  -> Fully Connected + Dropout
  -> Softmax class prediction
```

학습 결과는 `saved_models/wafer_cnn_model.pt`에 저장됩니다. 모델 파일이 아직 없을 때도 API와 대시보드는 rule-based baseline으로 예측을 수행할 수 있습니다.

## 6. DB 설계

SQLite 기본 경로: `data/wafer_quality.db`

```sql
CREATE TABLE wafer_info (
    wafer_id TEXT PRIMARY KEY,
    lot_id TEXT,
    failure_type TEXT,
    die_size INTEGER,
    inspection_date TEXT
);

CREATE TABLE prediction_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wafer_id TEXT,
    predicted_label TEXT,
    confidence REAL,
    model_name TEXT,
    created_at TEXT,
    FOREIGN KEY (wafer_id) REFERENCES wafer_info(wafer_id)
);

CREATE TABLE model_metric (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    trained_at TEXT
);

CREATE TABLE defect_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT,
    defect_type TEXT,
    defect_count INTEGER,
    defect_rate REAL,
    calculated_at TEXT
);
```

## 7. API 명세

FastAPI 실행:

```powershell
uvicorn src.api.main:app --reload --port 8000
```

엔드포인트:

| Method | Path | 설명 |
| --- | --- | --- |
| GET | `/health` | 서버 및 DB 상태 확인 |
| POST | `/predict` | wafer map 또는 wafer_id 기반 예측 |
| GET | `/results` | 최신 예측 결과 목록 |
| GET | `/results/{wafer_id}` | 특정 Wafer 예측 결과 |
| GET | `/metrics` | 모델 성능 지표 |
| GET | `/statistics/lot` | Lot별 불량률 |
| GET | `/statistics/defect` | Lot/불량유형별 통계 |

예측 요청 예시:

```json
{
  "wafer_id": "WAFER_000001"
}
```

응답 예시:

```json
{
  "wafer_id": "WAFER_000001",
  "predicted_label": "Edge-Ring",
  "confidence": 0.934,
  "model_name": "CNN_v1"
}
```

## 8. 대시보드 화면

Streamlit 실행:

```powershell
streamlit run src/dashboard/app.py
```

구성:

- Overview: 전체 wafer 수, 정상/불량 수, 불량률, 최빈 불량 유형
- Defect Distribution: 불량 유형별 bar/pie chart, Lot별 불량률
- Wafer Map Viewer: wafer map, 실제 label, 예측 label, confidence
- Model Performance: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Report: Lot별 통계, 모델 요약, Excel/PDF 다운로드

## 9. 성능 결과

전체 파이프라인 실행 후 `reports/model_metrics.json`과 `reports/confusion_matrix.csv`가 생성됩니다.

```powershell
python -m src.pipeline --epochs 6
```

현재 synthetic demo 데이터 기준 검증 결과:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.875 |
| Precision | 0.881 |
| Recall | 0.875 |
| F1-score | 0.865 |

실행 환경, random seed, epoch 수, 실제 데이터셋 교체 여부에 따라 결과는 달라질 수 있습니다.

## 10. 한계점 및 개선 방향

- 현재 synthetic 데이터는 포트폴리오 시연용입니다. 실제 WM-811K 등 wafer map 데이터셋으로 교체하면 모델 검증력이 높아집니다.
- CNN 모델은 baseline 구조입니다. ResNet 계열, class imbalance 보정, augmentation을 추가할 수 있습니다.
- SQLite는 로컬 데모에 적합합니다. 운영형 구조에서는 PostgreSQL, migration, 접근 권한 관리가 필요합니다.
- 모델 추론 API는 동기 방식입니다. 대량 inference에는 queue 기반 비동기 처리나 batch endpoint가 적합합니다.

## 실행 순서

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline --epochs 6
streamlit run src/dashboard/app.py
uvicorn src.api.main:app --reload --port 8000
```

Docker 실행:

```powershell
docker compose up --build
```
