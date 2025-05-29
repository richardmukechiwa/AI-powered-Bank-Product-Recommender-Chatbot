from BankProducts.config.configuration import ConfigurationManager
from BankProducts.components.data_validation import DataValidation
from BankProducts import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_gen_config = config.get_data_generation_config()
        
        # Validate CSV files
        data_validation = DataValidation(config=data_gen_config)
        data_validation.validate_file_exists(data_gen_config.customers_filename, "Customers CSV")
        data_validation.validate_csv_not_empty(data_gen_config.customers_filename, "Customers CSV")
        
        data_validation.validate_file_exists(data_gen_config.products_filename, "Products CSV")
        data_validation.validate_csv_not_empty(data_gen_config.products_filename, "Products CSV")
        
        
        # Validate database tables
        data_validation.validate_database_tables()
        
if __name__ == "__main__":
    logger.info(f"Starting {STAGE_NAME}")
    try:
        pipeline = DataValidationTrainingPipeline()
        pipeline.main()
        logger.info(f"{STAGE_NAME} completed successfully")
    except Exception as e:
        logger.exception(f"Error in {STAGE_NAME}: {e}")
        raise   