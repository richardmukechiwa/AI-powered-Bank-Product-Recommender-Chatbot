from BankProducts import logger
from BankProducts.entity.config_entity import FeatureImportanceConfig





class FeatureImportance:
    def __init__(self, config: FeatureImportanceConfig):
        self.config = config

    def important_feature(self):
        import joblib
        import shap
        import pandas as pd
        import numpy as np
        import os
        import matplotlib.pyplot as plt

        logger.info("Starting SHAP feature importance calculation...")

        # Load data
        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop(self.config.target_column, axis=1)

        # Load model and preprocessor
        pipeline = joblib.load(self.config.grid_search_model)
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['log_regression']

        # Transform data
        X_processed = preprocessor.transform(test_x)

        # Get feature names
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

        # SHAP Explainer
        explainer = shap.LinearExplainer(model, X_df, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_df)

        # Handle binary or multiclass
        # Handle binary or multiclass
        shap_abs = np.abs(shap_values)

        if shap_abs.ndim == 3:
            # Multiclass: (samples, features, classes) → average over classes
            shap_mean = shap_abs.mean(axis=2)
            shap_df = pd.DataFrame(shap_mean, columns=X_df.columns)
        else:
            shap_df = pd.DataFrame(shap_abs, columns=X_df.columns)

        shap_importance = shap_df.mean().sort_values(ascending=False)


        # Log top features
        print("Top Important Features:")
        print(shap_importance.head(10))

        # Create folder for plots
        plots_dir = self.config.feature_importance_file.parent / "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Bar Plot
        plt.figure(figsize=(10, 6))
        shap_importance.head(20).plot(kind='barh')
        plt.title("Top 20 Feature Importances (Mean SHAP Value)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        bar_plot_path = plots_dir / "shap_bar_plot.png"
        plt.savefig(bar_plot_path)
        plt.close()

        # 2. Beeswarm Plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_df, plot_type="dot", show=False)
        beeswarm_path = plots_dir / "shap_beeswarm_plot.png"
        plt.savefig(beeswarm_path, bbox_inches='tight')
        plt.close()

        # (Optional) 3. Force Plot for a single prediction
        # Uncomment to save interactive HTML
        # force_plot = shap.plots.force(explainer.expected_value, shap_values[0], X_df.iloc[0])
        # shap.save_html(str(plots_dir / "shap_force_plot.html"), force_plot)

        # Save SHAP importance to JSON
        shap_importance.to_json(self.config.feature_importance_file)
        logger.info(f"SHAP values saved to {self.config.feature_importance_file}")
        logger.info(f"SHAP bar plot saved to {bar_plot_path}")
        logger.info(f"SHAP beeswarm plot saved to {beeswarm_path}")

        return shap_importance




