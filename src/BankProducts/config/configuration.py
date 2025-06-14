from BankProducts.constants import *
from BankProducts.utils.common import read_yaml, create_directories
from BankProducts import logger
from BankProducts.entity.config_entity import (
                
                                        DataIngestionConfig,
                                        DataValidationConfig,
                                        DataTransformationConfig,
                                        ModelTrainingConfig,
                                        ModelEvaluationConfig,
                                        FeatureImportanceConfig,
                                        FinalModelConfig
                                        
                                        
                                        )

class ConfigurationManager: 
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH, 
        schema_filepath = SCHEMA_FILE_PATH,
        ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])
        
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
    
    
        create_directories([self.config.artifacts_root])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            #export_csv_path=Path(config.export_csv_path),
            output_path=Path(config.output_path),
            table_name=config.table_name
            
            
        )
        
        return data_ingestion_config
    
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([self.config.artifacts_root])

        data_validation_config = DataValidationConfig(
            val_root_dir=Path(config.val_root_dir),
            STATUS_FILE=Path(config.STATUS_FILE),
            customer_data=Path(config.customer_data),
            all_schema=schema,
        )

        return data_validation_config
    
    

    def get_data_transformation_config(self)-> DataTransformationConfig:
        """
        Returns Data Transformation Configuration
        """
        config = self.config.data_transformation
        schema =  self.schema.target_column
        
        create_directories([self.config.artifacts_root])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            transformed_data_file= Path(config.transformed_data_file),
            customer_path= Path(config.customer_path),
            train_data_file= Path(config.train_data_file),
            test_data_file= Path(config.test_data_file),
            target_column= schema.name
            )
        
        
        return data_transformation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params.logistic_regression
        schema =  self.schema.target_column

        create_directories([self.config.artifacts_root])

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            train_data_dir = Path(config.train_data_dir),
            test_data_dir = Path(config.test_data_dir),
            model_name = config.model_name,
            random_state = params.random_state,
            class_weight = params.class_weight,
            n_jobs = params.n_jobs,
            target_column = schema.name,
            label_encoder_file = config.label_encoder_file,
            C = params.C,
            penalty = params.penalty,
            solver = params.solver, 
            max_iter = params.max_iter
            )
           
            
        

        return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation   
        params = self.params.random_forest
        schema =  self.schema.target_column

        create_directories([self.config.artifacts_root])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metric_file_name=Path(config.metric_file_name),
            target_column=schema.name,
            params=params,
            grid_search_model_path= Path(config.grid_search_model_path),
            train_data_path=Path(config.train_data_path),
            encoded_target_label= Path(config.encoded_target_label)
            
            
           
            
        )

        return model_evaluation_config
            
           
           
    def get_feature_importance_config(self) -> FeatureImportanceConfig:
        config = self.config.feature_importance 
        schema = self.schema.target_column
       
        create_directories([self.config.artifacts_root])
       
        
        feature_importance_config = FeatureImportanceConfig(
            root_dir=Path(config.root_dir),
            grid_search_model=Path(config.grid_search_model),
            training_data_path=Path(config.training_data_path),
            test_data_path=Path(config.test_data_path),
            feature_importance_file=Path(config.feature_importance_file),
            target_column= schema.name
            
        )
        logger.info(f"Feature Importance Config: {feature_importance_config}")
        return feature_importance_config
        
        
    def get_final_model_config(self)-> FinalModelConfig:
        """Returns the final model configuration."""
        config = self.config.retrained_model
        params = self.params.logistic_regression
        schema = self.schema.target_column
        
        create_directories([self.config.artifacts_root])
        
        final_model_config = FinalModelConfig(
          
           target_column=schema.name,
           training_data= Path(config.training_data),
           testing_data= Path(config.testing_data),
           final_model= Path(config.final_model),
           penalty= params.penalty,
           C= params.C,
           max_iter= params.max_iter,
           random_state= params.random_state,
           solver= params.solver,
           n_jobs= params.n_jobs,
           model_encoder= Path(config.model_encoder),
           class_weight= params.class_weight,
           metric_file= Path(config.metric_file)
           
           
           
                         
            
        )
        
        
        
        return final_model_config   
        
    
    
    