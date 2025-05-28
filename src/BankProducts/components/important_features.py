from BankProducts import logger
from BankProducts.entity.config_entity import FeatureImportanceConfig





class FeatureImportance:
    def __init__(self, config: FeatureImportanceConfig):
        self.config = config
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.processor = None


    def important_features(self):
        
        """Compute and save SHAP feature importance for the model."""
        # Compute feature importances robustly for multiclass
        import joblib
        import shap
        import pandas as pd
        import numpy as np

        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop(self.config.target_column, axis=1)

        pipeline = joblib.load(self.config.grid_search_model)
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier']

        X_processed = preprocessor.transform(test_x)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            num_features = preprocessor.transformers_[0][2]
            cat_encoder = preprocessor.transformers_[1][1]
            cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
            feature_names = np.concatenate([num_features, cat_features])

        X_df = pd.DataFrame(
            X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed,
            columns=feature_names
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)

        # Handle multiclass
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            # Take mean absolute SHAP value across classes for each feature
            shap_abs = np.abs(shap_values).mean(axis=2)  # shape: (n_samples, n_features)
            shap_df = pd.DataFrame(shap_abs, columns=X_df.columns)
            shap_importance = shap_df.mean().sort_values(ascending=False)
        elif isinstance(shap_values, list) and isinstance(shap_values[0], np.ndarray):
            # shape: (n_classes, n_samples, n_features)
            shap_array = np.abs(np.array(shap_values))  # (n_classes, n_samples, n_features)
            shap_abs = shap_array.mean(axis=0)  # mean over classes -> (n_samples, n_features)
            shap_df = pd.DataFrame(shap_abs, columns=X_df.columns)
            shap_importance = shap_df.mean().sort_values(ascending=False)
        else:
            shap_df = pd.DataFrame(shap_values, columns=X_df.columns)
            shap_importance = shap_df.abs().mean().sort_values(ascending=False)

        # Print top important features
        
        print("Top Important Features:")
        print(shap_importance.head(10))

        # Optionally save to JSON
        import os
        os.makedirs(self.config.feature_importance_file.parent, exist_ok=True)
        shap_importance.to_json(self.config.feature_importance_file)

        logger.info(f"Feature importance saved to {self.config.feature_importance_file}")
        return shap_importance

