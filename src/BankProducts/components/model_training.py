import pandas as pd
import os
from BankProducts import logger
from sklearn.linear_model import LogisticRegression


import joblib
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from BankProducts.entity.config_entity import ModelTrainingConfig
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model = None

    def train(self):
        logger.info("Loading training data")
        
        os.makedirs(os.path.dirname(self.config.train_data_dir), exist_ok=True)
        # Load the training data
        
        train_data = pd.read_csv(self.config.train_data_dir)
        X_train = train_data.drop(columns=[self.config.target_column])
        
        
        y_train = train_data[self.config.target_column]
        
        # encode the target variable if it's categorical
        if y_train.dtype == 'object' or y_train.dtype.name == 'category':
            logger.info("Encoding target variable")
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
        else:
            logger.info("Target variable is already numeric, no encoding needed")   
        logger.info("Loading validation data")
        
         # Create directory if it doesn't exist
        model_dir = self.config.root_dir
        os.makedirs(model_dir, exist_ok=True)
        
       
        
        # save the label_encoder for future use
        label_encoder_path = os.path.join(model_dir , self.config.label_encoder_file)
        joblib.dump(label_encoder, label_encoder_path)
        
        logger.info("Label encoder saved to: %s", label_encoder_path)
         
        
        logger.info("Training the model")
        # Identify categorical and numerical features
        categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
        numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
      
        #preprocessor = ColumnTransformer(
            #transformers=[
                #('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                #('num', SimpleImputer(strategy='mean'), numerical_features)
            #],
           # remainder='passthrough'  # Keep other columns as they are
        #)

        # Initialize the RandomForestClassifier with the provided configuration
        model = LogisticRegression(
            C=self.config.C,
            penalty=self.config.penalty,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,      
            class_weight=self.config.class_weight,      
            n_jobs=self.config.n_jobs
        )
        #model = RandomForestClassifier(
        
        
       # Create the pipeline combining preprocessing, scaling, and modeling
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_features)
            ],
            remainder='passthrough' 
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Fit the pipeline to the training data
        
        
        print("X_TRAIN", X_train[:15])
        print("Y_train", y_train[:10])

        pipeline.fit(X_train, y_train)
        
        X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
        print("X_train after preprocessing:", X_train_transformed[:1])

        
        import numpy as np

        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']

        # Get feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()

        # Get the coefficient matrix (shape: n_classes x n_features)
        coefs = classifier.coef_

        # Take the mean of absolute values across all classes (axis=0)
        importances = np.mean(np.abs(coefs), axis=0)
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]

        # Limit to top N (avoid index errors)
        top_n = min(20, len(sorted_indices))

        # Display
        print("Top important features (averaged across all classes):")
        for i in sorted_indices[:top_n]:
            
            
            print(f"{feature_names[i]}: {importances[i]:.4f}")

        import matplotlib.pyplot as plt

        top_features = [feature_names[i] for i in sorted_indices[:top_n]]
        top_importance_values = [importances[i] for i in sorted_indices[:top_n]]

        plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_importance_values[::-1])
        plt.title("Top Features (Avg Absolute Coefficients - Multiclass Logistic Regression)")
        plt.xlabel("Average Absolute Coefficient")
        plt.tight_layout()
        plt.show()

        classes = classifier.classes_  # e.g., array([0, 1, 2])

        for class_index, class_label in enumerate(classes):
            print(f"\nTop features for class {class_label}:")
            class_coefs = coefs[class_index]
            sorted_idx = np.argsort(np.abs(class_coefs))[::-1]
            
            for i in sorted_idx[:10]:
                print(f"{feature_names[i]}: {class_coefs[i]:.4f}")



        
            
            import matplotlib.pyplot as plt

            top_features = [feature_names[i] for i in sorted_indices[:top_n]]
            top_importances = [importances[i] for i in sorted_indices[:top_n]]

            plt.figure(figsize=(10, 6))
            plt.barh(top_features[::-1], top_importances[::-1])
            plt.title("Top Important Features (Logistic Regression Coefficients)")
            plt.xlabel("Coefficient Value")
            plt.tight_layout()
            plt.show()

            


        logger.info("Saving the trained model")
        
                
        # Create directory if it doesn't exist
        model_dir = self.config.root_dir
        os.makedirs(model_dir, exist_ok=True)
        
        print("Model directory:", model_dir)
        print("Directory exists?", os.path.exists(model_dir))
        print("Is a directory?", os.path.isdir(model_dir))

                
        model_path = os.path.join(model_dir, self.config.model_name)
        
        print("Model path:", model_path)
        # Ensure the model path is a Path object
        model_path = Path(model_path)
        
        
        

        
        print("Model will be saved to:", model_path)
        
                # Safety check: if a directory exists where the model file should go, delete it
        if os.path.isdir(model_path):
            import shutil
            logger.warning(f"A folder exists at model path '{model_path}', deleting it.")
            shutil.rmtree(model_path)
            
        # Save the model to the specified path
        self.model = pipeline 

   
        # Save the model using joblib
        joblib.dump(self.model, model_path, compress=4)
      
        
        logger.info("Model training completed successfully")
        
        
        
        




