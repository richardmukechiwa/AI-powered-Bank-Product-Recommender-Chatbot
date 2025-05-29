from BankProducts.pipeline.stage_01_data_generation import DataGenerationTrainingPipeline
from BankProducts.pipeline.stage_02_data_ingestion import DataIngestionTrainingPipeline
from BankProducts.pipeline.stage_03_data_validation import DataValidationTrainingPipeline
from BankProducts.pipeline.stage_04_data_transformation import DataTransformationTrainingPipeline
from BankProducts.pipeline.stage_05_model_training import ModelTrainingPipeline
from BankProducts.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline
from BankProducts.pipeline.stage_07_importnant_features import FeatureImportancePipeline


STAGE_NAME = "Data Generation Stage"
try:
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

    print("Data Generation Complete")
except Exception as e:
    raise e