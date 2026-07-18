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


#Instead of two separate lists  predictions and actuals  that could get out of sync, each ForecastRecord holds both together, tied to the same timestamp.
#It also gives data validation: fields have to match their expected type, or it fails immediately instead of causing a bug downstream.
#Fields we don't have yet like actual_value  stay None until the real outcome is known, then get filled in.
#The schema version tracks the structure of the data itself. The version only bumps if the fields themselves change 
#and if they do, old records are still readable under their original schema.