#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/opt/anaconda3/bin/python}

"$PYTHON" -m pip install xgboost==3.1.3 mlflow==3.9.0 psutil==7.2.1

echo "Done. Run: RUN_INTEGRATION=1 $PYTHON -m pytest -q -m integration"
