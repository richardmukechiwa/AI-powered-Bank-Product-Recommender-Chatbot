from BankProducts.config.configuration import ConfigurationManager
from BankProducts.components.model_evaluation  import ModelEvaluation
from BankProducts import logger

STAGE_NAME =  "Mode Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        eval = ModelEvaluation(config=model_evaluation_config)
        eval.log_into_mlflow()
        eval.perform_grid_search() 
if __name__ == "__main__":
    try:
        model_evaluation = ModelEvaluationTrainingPipeline()
        model_evaluation.main()
        logger.info(" Model Evaluation Stage completed successfully")
    except Exception as e:
        logger.error(f" Model Evaluation Stage failed with error {str(e)}")
        raise e
            