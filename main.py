import os
from BankProducts.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from BankProducts.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from BankProducts.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from BankProducts.pipeline.stage_04_model_training import ModelTrainingPipeline
from BankProducts.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from BankProducts.pipeline.stage_06_important_features import FeatureImportanceTrainingPipeline
from BankProducts.pipeline.stage_07_final_model import FinalModelTrainingPipeline
from BankProducts import logger


STAGE_NAME = "Data Ingestion Stage"
try:
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logger.info(f"{STAGE_NAME} completed successfully")
except Exception as e:
    logger.exception(f"Error in {STAGE_NAME}: {e}")
    raise

STAGE_NAME = "Data Validation Stage"
try:
    pipeline = DataValidationTrainingPipeline()
    pipeline.main()
    logger.info(f"{STAGE_NAME} completed successfully")
except Exception as e:
    logger.exception(f"Error in {STAGE_NAME}: {e}")
    raise  
 
STAGE_NAME = "Data Transformation Stage"
try:
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()
    logger.info(f"Data Transformation Stage completed successfully. {STAGE_NAME}")  
except Exception as e:
    logger.exception(f"Data Transformation Stage failed. {STAGE_NAME}") 
    raise e

STAGE_NAME = "Model Training Stage"
try:
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logger.info(f"Model Training Stage completed successfully")
except Exception as e:
    logger.error(f"Model Training Stage failed with error: {str(e)}")
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(" Model Evaluation Stage completed successfully")
except Exception as e:
    logger.error(f" Model Evaluation Stage failed with error {str(e)}")
    raise e

STAGE_NAME = "Feature Importance Stage"
try:
    pipeline  = FeatureImportanceTrainingPipeline()
    pipeline.main()
    logger.info(" Important Features Analysis completed")
except Exception as e:
    logger.error(f"Error in Important Features Analysis: {str(e)}")
    raise e

STAGE_NAME = "Final Model"
try:
    pipeline = FinalModelTrainingPipeline()
    pipeline.main()
    logger.info(f"Final Model Training Stage completed successfully")
except Exception as e:
    raise e
        
            
    
