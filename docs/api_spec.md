# API Spec

Base URL: `http://localhost:8000`

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

## `GET /results`

Query parameters:

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `limit` | integer | `100` | Number of latest wafer predictions |

## `GET /results/{wafer_id}`

Returns the latest prediction for one wafer.

## `GET /metrics`

Returns latest metric and metric history.

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
