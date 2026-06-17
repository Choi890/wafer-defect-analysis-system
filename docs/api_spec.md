# API Spec

Base URL: `http://localhost:8000`

## `GET /health`

Returns service status and whether the local database file exists.

```json
{
  "status": "ok",
  "database_ready": true
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

## `GET /statistics/lot`

Returns total wafer count, defect count, and defect rate per lot.

## `GET /statistics/defect`

Returns defect count and defect rate per lot and defect type.
