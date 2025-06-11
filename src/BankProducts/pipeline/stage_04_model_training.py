from BankProducts.config.configuration import ConfigurationManager
from BankProducts.components.model_training import ModelTraining
from BankProducts import logger

STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_trainer = ModelTraining(config=model_training_config)
        model_trainer.train()
            

if __name__ == "__main__":
    try:
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f"Model Training Stage completed successfully")
    except Exception as e:
        logger.error(f"Model Training Stage failed with error: {str(e)}")
        raise e

