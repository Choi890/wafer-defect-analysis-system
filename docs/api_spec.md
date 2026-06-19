# API Spec

Base URL: `http://localhost:8000`

If `WAFER_API_KEY` is set, include it on every request:

```text
x-api-key: <your-api-key>
```

## `GET /health`

Returns service status and whether the local database file exists.

```json
{
  "status": "ok",
  "database_ready": true,
  "model_ready": true,
  "wafer_count": 360,
  "prediction_count": 360,
  "version": "0.2.0",
  "environment": "local"
}
```

## `POST /predict`

Predicts a wafer defect class. Provide either an existing `wafer_id` or a custom `wafer_map`.

Request using existing wafer:

```json
{
  "wafer_id": "WAFER_000001"
}
```

Request using explicit wafer map:

```json
{
  "wafer_id": "API_SAMPLE_001",
  "lot_id": "LOT_API",
  "wafer_map": [[0, 0, 1], [0, 2, 1], [0, 1, 1]]
}
```

Response:

```json
{
  "wafer_id": "WAFER_000001",
  "predicted_label": "Center",
  "confidence": 0.934,
  "model_name": "CNN_v1"
}
```

## `POST /predict/batch`

Queues a batch prediction job. This uses an in-process background worker for local/PoC deployments and persists job status in SQLite.

```json
{
  "items": [
    {"wafer_id": "WAFER_000001"},
    {"wafer_id": "API_BATCH_001", "lot_id": "LOT_API", "wafer_map": [[0, 1], [1, 2]]}
  ]
}
```

Response:

```json
{
  "job_id": "JOB_ABC123DEF456",
  "status": "queued",
  "requested_count": 2
}
```

## `GET /jobs`

Returns recent batch prediction jobs.

## `GET /jobs/{job_id}`

Returns one batch prediction job status.

## `GET /results`

Query parameters:

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `limit` | integer | `100` | Number of latest wafer predictions |

## `GET /results/{wafer_id}`

Returns the latest prediction for one wafer.

## `GET /metrics`

Returns latest metric and metric history.

The latest metric includes aggregate scores plus training metadata when the enhanced pipeline has run:

```json
{
  "model_name": "CNN_v1",
  "accuracy": 0.9028,
  "precision_score": 0.9031,
  "recall_score": 0.9028,
  "f1_score": 0.8908,
  "macro_f1_score": 0.8908,
  "best_validation_f1": 0.8775,
  "best_epoch": 6,
  "device": "cuda"
}
```

## `GET /summary`

Returns executive quality KPIs for the dashboard.

```json
{
  "total_wafers": 360,
  "total_lots": 15,
  "defect_wafers": 320,
  "defect_rate": 0.8889,
  "top_defect": "Center",
  "risk_lots": 13,
  "critical_lots": 13,
  "avg_confidence": 0.7975,
  "low_confidence_count": 78,
  "model_f1": 0.8654
}
```

## `GET /statistics/lot`

Returns total wafer count, defect count, and defect rate per lot.

## `GET /statistics/defect`

Returns defect count and defect rate per lot and defect type.

## `GET /models`

Returns the local model registry from `saved_models/model_registry.json`.

## `GET /monitoring`

Returns an operations snapshot including service readiness, quality summary, latest metric, recent batch jobs, and model registry count.

## `GET /monitoring/prometheus`

Returns a Prometheus-compatible text payload for core gauges such as wafer count, defect rate, risk lots, low-confidence predictions, and latest F1.
