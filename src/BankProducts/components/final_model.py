from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from BankProducts import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score,
                             precision_score, 
                             recall_score, f1_score,
                             confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay,
                            )
from urllib.parse import urlparse
import tempfile
from BankProducts.utils.common import save_json
import os 
from BankProducts.entity.config_entity import FinalModelConfig
from pathlib import Path


class FinalModel:
    def __init__(self, config: FinalModelConfig):
        self.config = config
        
    def final_training(self):
        #make sure the path exists
        os.makedirs(os.path.dirname(self.config.training_data), exist_ok=True)
        
        df = pd.read_csv(self.config.training_data)
        
        # Correct column selection using a list
        data = df[["monthlyincome", "productcategory", "most_used_channel", 
                   "productsubcategory", "amount", "recommendedoffer"]]
        
        # Drop target column to get training features
        X_train = data.drop(columns=[self.config.target_column])
        Y_train = data[self.config.target_column]
        
        print(data.head())
        print(X_train.head())
        
        print(Y_train.head())
        
        # apply Label Encoder to categorical variables
        le = LabelEncoder()
        Y_train = le.fit_transform(Y_train)
        
        # make sure the path exists
        os.makedirs(os.path.dirname(self.config.model_encoder), exist_ok=True)
        # save the Label Encoder  using joblib
        joblib.dump(le, self.config.model_encoder)
        
        
        
        # apply OneHot Encoder to categorical variables and scale numerical variables 
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        numerical_cols = X_train.select_dtypes(include=['int64']).columns
        #print(categorical_cols)
        #print(numerical_cols)
        
        # define the Logistic Regression model 
        model = LogisticRegression( C= self.config.C,
                                   penalty = self.config.penalty,
                                   solver = self.config.solver,
                                   max_iter = self.config.max_iter,
                                   random_state = self.config.random_state,
                                   n_jobs= self.config.n_jobs
                                   )
        
        # Create the ColumnTransformer and  pipeline combining preprocessing, scaling, and modeling
        preprocessor = ColumnTransformer( transformers = [
            ('num', StandardScaler(), numerical_cols),  
            ('cat', OneHotEncoder(handle_unknown= 'ignore', sparse_output= False), categorical_cols)  
            ],
                                         remainder='passthrough',
                                         n_jobs = self.config.n_jobs        
                                         )
        
        # create the pipeline
        pipeline = Pipeline(steps = [
            ('preprocessor', preprocessor),
            ('model', model)
            ]   
        )
        
        # fit the pipeline to the training data
        pipeline.fit(X_train, Y_train)  
        
        # make sure the path exists
        os.makedirs(os.path.dirname(self.config.final_model), exist_ok=True)
        
        # save the pipeline using joblib
        joblib.dump(pipeline, self.config.final_model)
        
        return data, X_train, Y_train
    
    #def model_evaluation(self):
        
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, precision, recall, f1

    def log_confusion_matrix(self, actual, predicted, class_names):
        cm = confusion_matrix(actual, predicted)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        temp_img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plt.savefig(temp_img_path)
        plt.close()
        mlflow.log_artifact(temp_img_path, artifact_path="confusion_matrix")

    def log_classification_report(self, actual, predicted, class_names):
        report = classification_report(actual, predicted, target_names=class_names)
        temp_txt_path = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
        with open(temp_txt_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(temp_txt_path, artifact_path="bank_products_recommender")

    def log_into_mlflow(self):
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        mlflow.set_experiment("Product Recommender")

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run():
            #load the model and the data
            #make sure the path exists
            os.makedirs(os.path.dirname(self.config.testing_data), exist_ok=True)
            
            test_df = pd.read_csv(self.config.testing_data)
            
            # Correct column selection using a list
            data = test_df[["monthlyincome", "productcategory", "most_used_channel", 
                    "productsubcategory", "amount", "recommendedoffer"]]
            
            
            X_test = data.drop(columns=self.config.target_column)
            Y_test = data[self.config.target_column]
            
            print(data.head())
            print(X_test.head())
            
            print(Y_test.head())
            
            # Encode the target variable
            le = joblib.load(self.config.model_encoder)
            test_y_encoded = le.transform(Y_test)
            
            # load the model
            pipeline = joblib.load(self.config.final_model)
            predicted = pipeline.predict(X_test)

            accuracy, precision, recall, f1 = self.eval_metrics(test_y_encoded, predicted)
            
             # evaluate the model
            rf_report = classification_report(test_y_encoded, predicted)
            rf_cm = confusion_matrix(test_y_encoded, predicted)   
            rf_accuracy = accuracy_score(test_y_encoded, predicted)   
            
            
            #create Confusion Matrix Display
            plt.figure(figsize=(15, 7))
            ax = plt.gca()

            cm_display = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=le.classes_)

            cm_display.plot(ax=ax)

            plt.title("Classifier Matrix")

            # Rotate x-axis labels to 45 degrees for better readability
            plt.xticks(rotation=45, ha='right')  

            # Rotate y-axis labels to 0 degrees (horizontal) for clarity
            plt.yticks(rotation=0)

            plt.tight_layout()  # Adjust layout to fit everything nicely

            plt.show()

            
            

            logger.info(f"Classification Report:\n{rf_report}")
            logger.info(f"Confusion Matrix:\n{rf_cm}") 
            logger.info(f"sAccuracy: {rf_accuracy}")
            

            scores = {
                "model_name": "logistic_classifier",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
            # Ensure directory exists
            Path(self.config.metric_file).parent.mkdir(parents=True, exist_ok=True)

            save_json(Path(self.config.metric_file), data=scores)
            
            logger.info("Metrics saved to: %s", self.config.metric_file)
            
            logger.info("Logging accuracy")
            mlflow.log_metric("accuracy", accuracy)

            logger.info("Logging precision")
            mlflow.log_metric("precision", precision)

            logger.info("Logging recall")
            mlflow.log_metric("recall", recall)

            logger.info("Logging f1_score")
            mlflow.log_metric("f1_score", f1)

            logger.info("Setting class_names")
            class_names = le.classes_

            logger.info("Logging confusion matrix")
            self.log_confusion_matrix(test_y_encoded, predicted, class_names)

            logger.info("Logging classification report")
            self.log_classification_report(test_y_encoded, predicted, class_names)
            
            
            logger.info("Tracking URI scheme: %s", tracking_url_type_store)
            logger.info("Tracking URI: %s", mlflow.get_tracking_uri())


            #if tracking_url_type_store != "file":
                #mlflow.sklearn.log_model(pipeline, "pipeline", registered_model_name="product recommender")
            #else:
                #mlflow.sklearn.log_model(pipeline, "pipeline")
              
            logger.info("mlflow model logged successfully.")
            
            return accuracy, precision, recall, f1, scores, class_names, 
        
        
        
        
        
        
        
        
        
        
        
        
        
