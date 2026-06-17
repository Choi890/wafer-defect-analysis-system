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
WAFER_MODEL_PATH=saved_models/wafer_cnn_model.pt
WARNING_DEFECT_RATE=0.70
CRITICAL_DEFECT_RATE=0.85
LOW_CONFIDENCE_THRESHOLD=0.70
```

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

## 운영 전환 시 보강 포인트

- SQLite를 PostgreSQL로 교체
- Alembic 기반 DB migration 추가
- API 인증 및 권한 관리 추가
- 모델 registry 또는 object storage 연동
- 배치 추론 작업 queue 도입
- Prometheus/Grafana 기반 모니터링 추가
- 실제 Wafer Map 데이터셋으로 재학습 및 검증
