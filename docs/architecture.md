# Architecture

```mermaid
flowchart TD
    A[Raw Wafer Map Data] --> B[Preprocessing Module]
    B --> C[CNN Training / Prediction]
    C --> D[(SQLite Database)]
    D --> E[FastAPI Backend]
    D --> F[Streamlit Dashboard]
    F --> G[Excel / PDF Reports]
    E --> H[Health / Summary API]

    B --> B1[Resize Wafer Map]
    B --> B2[Label Encoding]
    B --> B3[Train/Test Split]

    D --> D1[wafer_info]
    D --> D2[prediction_result]
    D --> D3[model_metric]
    D --> D4[defect_statistics]
```

## Data Lifecycle

1. `src.data.load_data` loads `data/raw/wafer_map_dataset.pkl` or creates synthetic wafer map records.
2. `src.data.preprocess` normalizes wafer map size, saves `.npy` maps, writes train/test CSV files, and writes `label_map.json`.
3. `src.models.train` trains the CNN, evaluates the test set, stores metrics, and writes predictions to SQLite.
4. `src.api.main` exposes prediction, health, summary, metric, and statistics endpoints.
5. `src.dashboard.app` reads from SQLite and renders an operations dashboard.
6. `src.utils.report_generator` exports Excel/PDF reports.

## Operations Dashboard

The dashboard is organized around production-like monitoring views:

- Control Tower: KPI summary, risk lots, low-confidence predictions
- Lot Analytics: lot-level defect rate and defect distribution
- Wafer Review: wafer map inspection and prediction detail
- Model Ops: model metric, confidence distribution, confusion matrix, audit log
- Reports: Excel/PDF generation and executive snapshot

## ERD

```mermaid
erDiagram
    wafer_info ||--o{ prediction_result : has
    wafer_info {
        text wafer_id PK
        text lot_id
        text failure_type
        integer die_size
        text inspection_date
    }
    prediction_result {
        integer id PK
        text wafer_id FK
        text predicted_label
        real confidence
        text model_name
        text created_at
    }
    model_metric {
        integer id PK
        text model_name
        real accuracy
        real precision_score
        real recall_score
        real f1_score
        text trained_at
    }
    defect_statistics {
        integer id PK
        text lot_id
        text defect_type
        integer defect_count
        real defect_rate
        text calculated_at
    }
```
