import joblib
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("artifacts/retrained_model/fin_model.joblib"))
        
        
        
    def predict(self, data):
        prediction = self.model.predict(data)

        # If you need to inverse transform labels (e.g., LabelEncoder)
        le = joblib.load(
            Path("artifacts/retrained_model/labelencorder.joblib")
        )
        if le is not None:
            outcome = le.inverse_transform(prediction)
        else:
            outcome = prediction

        return outcome

        
        