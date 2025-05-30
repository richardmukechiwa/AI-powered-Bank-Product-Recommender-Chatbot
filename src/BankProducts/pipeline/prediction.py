import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("artifacts/model_training/grid_search_model.joblib"))
        
        
        
    def predict(self, data):
        prediction = self.model.predict(data)

        # If you need to inverse transform labels (e.g., LabelEncoder)
        if hasattr(self, 'le'):
            outcome = self.le.inverse_transform(prediction)
        else:
            outcome = prediction

        return outcome

        
        