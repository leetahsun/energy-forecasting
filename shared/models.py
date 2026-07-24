from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ForecastRecord(BaseModel):
    """A single forecast prediction, paired with the actual value once
    it becomes known, so evaluate.py can compute accuracy over time
    without a separate join step.
    """

    timestamp: datetime
    metric_name: str          # "price_eur_mwh", "renewable_share_pct", "solar_mw"
    model_name: str           # "xgboost_v1", "naive_baseline", "clearsky_physical"
    predicted_value: float
    actual_value: Optional[float] = None
    segment: str = "DE"
    schema_version: str = "1.0"
