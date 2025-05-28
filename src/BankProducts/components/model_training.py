import pandas as pd
import os
from BankProducts import logger
from sklearn.ensemble import RandomForestClassifier


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
        
        
        logger.info("Training the model")
        # Identify categorical and numerical features
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
      
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                ('num', SimpleImputer(strategy='mean'), numerical_features)
            ],
            remainder='passthrough'  # Keep other columns as they are
        )

        # Initialize the RandomForestClassifier with the provided configuration
        model = RandomForestClassifier(
            criterion=self.config.criterion,
            max_features=self.config.max_features,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            n_estimators=self.config.n_estimators,  
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,
            n_jobs=self.config.n_jobs
        )
        

        # Create the pipeline combining preprocessing, scaling, and modeling
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),   # Optional: Only applies to numerical after preprocessing
            ('classifier', model)
        ])
        
        # Fit the pipeline to the training data

        pipeline.fit(X_train, y_train)

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



