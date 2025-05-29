from BankProducts.config.configuration import ConfigurationManager
from BankProducts.components.data_generation import DataGeneration
from BankProducts import logger

STAGE_NAME = "Data Generation Stage"

class DataGenerationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_generation_config = config.get_data_generation_config()
        data_gen = DataGeneration(config=data_generation_config)

        # Generate data
        customers_df, products_df = data_gen.generate_customer_data()

        # Save to CSV
        customers_path, products_path = data_gen.save_to_csv(customers_df, products_df, data_generation_config.output_dir)
        logger.info(f"Customers data saved to {customers_path}")

        # Save to DB
        data_gen.save_to_db(customers_path, products_path, data_generation_config.db_file)
        logger.info(f"Data saved to SQLite database at {data_generation_config.db_file}")

if __name__ == "__main__":
    logger.info(f"Starting {STAGE_NAME}")
    try:
        pipeline = DataGenerationTrainingPipeline()
        pipeline.main()
        logger.info(f"{STAGE_NAME} completed successfully")
    except Exception as e:
        logger.exception(f"Error in {STAGE_NAME}: {e}")
        raise
