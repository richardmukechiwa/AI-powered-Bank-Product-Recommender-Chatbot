from BankProducts.config.configuration import ConfigurationManager
from BankProducts.components.final_model import FinalModel
from BankProducts import logger

STAGE_NAME =  "Final Model"  # Name of the stage

class FinalModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        final_model_config = config.get_final_model_config()
        fin_model = FinalModel(config = final_model_config)

        fin_model.final_training()
        fin_model.log_into_mlflow()
if __name__ == "__main__":
    try:
        pipeline = FinalModelTrainingPipeline()
        pipeline.main() 
        logger.info(f" {STAGE_NAME} stage finished successfully")
    except Exception as e:
        logger.exception(e)
        
        

        