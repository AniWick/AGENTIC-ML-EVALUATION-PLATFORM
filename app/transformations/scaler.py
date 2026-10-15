import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataScaler:
    """Scales numerical features using Standard or MinMax normalization."""

    @staticmethod
    def scale_data(df: pd.DataFrame, method: str = "standard", target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_scaled = df.copy()
        target = target_column or df_scaled.columns[-1]
        num_cols = [c for c in df_scaled.select_dtypes(include=[np.number]).columns if c != target]

        if not num_cols:
            return df_scaled, {"scaled_features": []}

        if method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols].fillna(0))
        return df_scaled, {"method": method, "scaled_features": num_cols}

data_scaler = DataScaler()

