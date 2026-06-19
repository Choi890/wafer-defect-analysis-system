# 운영 가이드

## 실행 구성

이 프로젝트는 로컬 데모, 사내 PoC, 컨테이너 배포를 모두 고려한 구조입니다.

| 구성 | 역할 |
| --- | --- |
| Streamlit | 품질 관제 대시보드 |
| FastAPI | 예측, 지표, 통계 API |
| SQLite | 로컬/PoC 저장소 |
| PyTorch | Wafer Map 불량 분류 모델 |
| Excel/PDF Report | Lot별 품질 리포트 |

## 환경 변수

`.env.example`을 기준으로 운영 환경에 맞게 값을 설정합니다.

```text
APP_ENV=local
WAFER_DATABASE_PATH=data/wafer_quality.db
WAFER_RAW_DATA_PATH=data/raw/wafer_map_dataset.pkl
WAFER_ENABLE_SYNTHETIC_DATA=true
WAFER_MODEL_PATH=saved_models/wafer_cnn_model.pt
WAFER_MODEL_REGISTRY_PATH=saved_models/model_registry.json
WAFER_API_KEY=
WARNING_DEFECT_RATE=0.70
CRITICAL_DEFECT_RATE=0.85
LOW_CONFIDENCE_THRESHOLD=0.70
```

`WAFER_RAW_DATA_PATH`는 `.pkl`, `.csv`, `.json`, `.jsonl` 파일을 지원합니다. CSV/JSON 파일은 `wafer_id`, `lot_id`, `failure_type`, 그리고 `wafer_map` 또는 `wafer_map_path`를 포함해야 합니다. `wafer_map`은 `[[0,1],[1,2]]` 형태의 2D JSON 문자열을 사용할 수 있고, `wafer_map_path`는 `.npy` 파일 경로를 사용할 수 있습니다.

공유 환경이나 배포 환경에서는 `WAFER_API_KEY`를 설정합니다. 값이 설정되면 API 요청에 `x-api-key` 헤더가 필요합니다.

## 품질 기준

| 기준 | 기본값 | 설명 |
| --- | ---: | --- |
| Warning 불량률 | 70% | Lot 품질 확인 필요 |
| Critical 불량률 | 85% | 우선 조치 대상 |
| 낮은 신뢰도 | 70% | 모델 재검토 또는 수동 검토 대상 |

대시보드 사이드바에서 임시 조정할 수 있고, 서버 환경 변수로 기본값을 변경할 수 있습니다.

## 헬스체크

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

응답에는 DB 준비 상태, 모델 파일 상태, wafer 수, prediction 수, 앱 버전, 환경명이 포함됩니다.

Prometheus 형식의 핵심 지표는 다음 endpoint에서 확인합니다.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/monitoring/prometheus
```

배치 예측 job 상태는 `/jobs`와 `/jobs/{job_id}`에서 확인합니다.

## 모델 재학습과 성능 점검

```powershell
python -m src.pipeline --epochs 6
```

재학습 시 다음 산출물이 갱신됩니다.

| 파일 | 내용 |
| --- | --- |
| `saved_models/wafer_cnn_model.pt` | validation F1 기준 best checkpoint |
| `saved_models/model_registry.json` | 모델 등록 이력 |
| `reports/model_metrics.json` | 전체 성능 지표 |
| `reports/classification_report.csv` | class별 precision/recall/F1/support |
| `reports/training_history.csv` | epoch별 train loss, validation loss, validation F1 |
| `reports/confusion_matrix.csv` | test-set confusion matrix |

대시보드의 `Model Ops` 탭에서 학습 이력, class별 성능, confusion matrix, 예측 audit log를 함께 확인합니다.

## 운영 전환 시 보강 포인트

- PostgreSQL 전환 또는 관리형 DB 도입
- Alembic 기반 DB migration 도입
- 역할 기반 권한 관리 추가
- 원격 model registry 또는 object storage 연동
- 외부 queue/Celery/RQ 기반 배치 추론 작업자 도입
- Prometheus/Grafana 기반 모니터링 추가
- 실제 Wafer Map 데이터셋으로 재학습 및 검증
- Autoencoder 기반 minority-class augmentation 또는 handcrafted spatial feature fusion 검토
