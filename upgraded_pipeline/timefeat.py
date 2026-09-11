"""Time-of-day features, identical to Try_increase_accuracy/common.py."""
import numpy as np
import pandas as pd


def hour_features(windows):
    """Local hour of day as (sin, cos). ExtraSensory was recorded around UC San
    Diego; window ids are UTC epochs, converted with daylight saving."""
    t = pd.to_datetime(np.asarray(windows), unit="s", utc=True).tz_convert("America/Los_Angeles")
    h = (t.hour + t.minute / 60.0).to_numpy(dtype=np.float64)
    ang = 2 * np.pi * h / 24.0
    return np.column_stack([np.sin(ang), np.cos(ang)]).astype(np.float32)
