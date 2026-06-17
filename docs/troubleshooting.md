# Troubleshooting

## Pandas import error with NumPy 2

If the global Python environment has an ABI mismatch such as `numpy.core.multiarray failed to import`, create a project virtual environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The project pins `numpy<2.0` to avoid common binary compatibility issues with older compiled packages.

## No real wafer dataset

The project automatically creates synthetic wafer maps when `data/raw/wafer_map_dataset.pkl` does not exist.

```powershell
python -m src.data.load_data
```

## Dashboard has no trained model metric

Run the full pipeline:

```powershell
python -m src.pipeline --epochs 6
```

The dashboard still works before training by using the rule-based prediction fallback.

## FastAPI cannot import `src`

Run commands from the project root:

```powershell
cd wafer-defect-analysis-system
uvicorn src.api.main:app --reload
```

## Rebuild demo data from scratch

```powershell
python -m src.pipeline --force-data --epochs 6
```
