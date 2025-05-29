from BankProducts.config.configuration import ConfigurationManager
from BankProducts.components.data_transformation import DataTransformation
from BankProducts import logger


STAGE_NAME= "Data Transformation Stage"

class DataTransformationTrainingPipeline():
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        
        data_transformation = DataTransformation(config=data_transformation_config)
        
        data_transformation.join_datasets()
        data_transformation.transform_data()
        data_transformation.split_data()
        
if __name__ == "__main__":
    try:
        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()
        logger.info(f"Data Transformation Stage completed successfully. {STAGE_NAME}")  
    except Exception as e:
        logger.exception(f"Data Transformation Stage failed. {STAGE_NAME}") 
        raise e
    


