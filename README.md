# Wafer Defect Analysis System

반도체 Wafer 불량 분석 및 품질 모니터링 시스템입니다. Wafer Map 데이터를 전처리하고 CNN 기반 모델로 불량 패턴을 분류한 뒤, 분석 결과를 DB에 저장하고 FastAPI와 Streamlit 대시보드로 제공하는 end-to-end 제조 품질 분석 프로젝트입니다.

## 주요 기능

- Wafer Map 데이터 생성/로드 및 전처리
- `Normal`, `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch` 분류
- PyTorch CNN 모델 학습, 평가, 저장
- SQLite 기반 분석 결과 저장
- FastAPI 예측/통계/헬스체크 API
- Streamlit 품질 관제 대시보드
- Lot별 품질 리스크, 예측 신뢰도, Confusion Matrix 모니터링
- Excel/PDF 자동 리포트 생성
- Docker 실행 구성

## 상용화형 업그레이드 포인트

현재 버전은 단순 데모 화면이 아니라 운영형 품질 모니터링 시스템처럼 보이도록 개선되어 있습니다.

| 영역 | 개선 내용 |
| --- | --- |
| Dashboard | Control Tower, Lot Analytics, Wafer Review, Model Ops, Reports 탭 구성 |
| 운영 KPI | 전체 wafer 수, Lot 수, 불량률, 위험 Lot, 평균 신뢰도, 모델 F1 표시 |
| 품질 리스크 | Warning/Critical 불량률 기준으로 Lot 위험도 분류 |
| 모델 운영 | 예측 신뢰도 분포, 낮은 신뢰도 예측, audit log 제공 |
| API 운영성 | `/health`, `/summary`, 요청 ID, 처리 시간 헤더, CORS 추가 |
| 설정 | `.env.example` 기반 환경 변수 설정 지원 |
| 문서 | API spec, architecture, operations guide, troubleshooting 제공 |

## 프로젝트 구조

```text
wafer-defect-analysis-system/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── docs/
│   ├── api_spec.md
│   ├── architecture.md
│   ├── operations.md
│   └── troubleshooting.md
├── notebooks/
├── reports/
├── saved_models/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   ├── database/
│   ├── models/
│   └── utils/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 실행 방법

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline --epochs 6
```

대시보드 실행:

```powershell
streamlit run src/dashboard/app.py
```

API 실행:

```powershell
uvicorn src.api.main:app --reload --port 8000
```

Docker 실행:

```powershell
docker compose up --build
```

## 대시보드 구성

| 탭 | 내용 |
| --- | --- |
| Control Tower | 핵심 KPI, Lot 위험도, 불량 Pareto, 낮은 신뢰도 예측 |
| Lot Analytics | Lot별 불량률, 불량 유형 분포, Lot 품질 테이블 |
| Wafer Review | Wafer Map 시각화, 실제/예측 라벨, 예측 상세 |
| Model Ops | 모델 성능, 신뢰도 분포, Confusion Matrix, 예측 audit log |
| Reports | Excel/PDF 리포트 생성 및 다운로드 |

## API

| Method | Path | 설명 |
| --- | --- | --- |
| GET | `/health` | 앱, DB, 모델 상태 확인 |
| GET | `/summary` | 품질 KPI 요약 |
| POST | `/predict` | wafer_id 또는 wafer_map 기반 예측 |
| GET | `/results` | 최신 예측 결과 목록 |
| GET | `/results/{wafer_id}` | 특정 wafer 예측 결과 |
| GET | `/metrics` | 모델 성능 지표 |
| GET | `/statistics/lot` | Lot별 불량률 |
| GET | `/statistics/defect` | Lot/불량 유형별 통계 |

API 문서:

```text
http://127.0.0.1:8000/docs
```

## 환경 변수

`.env.example`:

```text
APP_ENV=local
WAFER_DATABASE_PATH=data/wafer_quality.db
WAFER_MODEL_PATH=saved_models/wafer_cnn_model.pt
WARNING_DEFECT_RATE=0.70
CRITICAL_DEFECT_RATE=0.85
LOW_CONFIDENCE_THRESHOLD=0.70
```

## 데이터와 모델

실제 wafer 데이터가 없으면 `data/raw/wafer_map_dataset.pkl` synthetic demo 데이터가 자동 생성됩니다. 실제 공개 데이터셋이나 보유 데이터를 사용할 경우 동일한 컬럼 구조로 변환한 뒤 raw dataset을 교체하면 됩니다.

입력 데이터 컬럼:

| 컬럼 | 설명 |
| --- | --- |
| `wafer_id` | Wafer 식별자 |
| `lot_id` | Lot 식별자 |
| `wafer_map` | 2D wafer map 배열 |
| `failure_type` | 실제 불량 유형 |
| `die_size` | die 개수 |
| `inspection_date` | 검사 일자 |

## 성능 결과

Synthetic demo 데이터 기준 검증 결과:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.875 |
| Precision | 0.881 |
| Recall | 0.875 |
| F1-score | 0.865 |

실행 환경, epoch 수, 실제 데이터셋 교체 여부에 따라 결과는 달라질 수 있습니다.

## 운영 전환 시 개선 방향

- SQLite를 PostgreSQL로 교체
- Alembic migration 추가
- API 인증과 권한 관리 추가
- 모델 registry 또는 object storage 연동
- 배치 추론 queue 도입
- Prometheus/Grafana 모니터링 추가
- 실제 Wafer Map 데이터 기반 재학습 및 검증
