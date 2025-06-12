from BankProducts.config.configuration import ConfigurationManager
from BankProducts.components.important_features import FeatureImportance
from BankProducts import logger

STAGE_NAME = "Important Feature Analysis"

class FeatureImportanceTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        feature_importance_config = config.get_feature_importance_config()
        feature_imp = FeatureImportance(config = feature_importance_config)
        feature_imp.important_feature()
        
if __name__ == "__main__":
    try:
        pipeline  = FeatureImportanceTrainingPipeline()
        pipeline.main()
        logger.info(" Important Features Analysis completed")
    except Exception as e:
        logger.error(f"Error in Important Features Analysis: {str(e)}")
        raise e
        
            
        
        
        
        