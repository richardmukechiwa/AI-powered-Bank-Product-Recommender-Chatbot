from BankProducts.config.configuration import ConfigurationManager
from BankProducts.components.data_validation import DataValidation
from BankProducts import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        
        # Validate CSV files
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()
        print("Data validation completed successfully.")
        
   
        
        
if __name__ == "__main__":
    logger.info(f"Starting {STAGE_NAME}")
    try:
        pipeline = DataValidationTrainingPipeline()
        pipeline.main()
        logger.info(f"{STAGE_NAME} completed successfully")
    except Exception as e:
        logger.exception(f"Error in {STAGE_NAME}: {e}")
        raise   