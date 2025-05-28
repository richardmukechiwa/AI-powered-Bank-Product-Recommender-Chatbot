from BankProducts.constants import *
from BankProducts.utils.common import read_yaml, create_directories
from BankProducts.utils.logger import logger
from BankProducts.entity.config_entity import (
                                        DataGenerationConfig,
                                        DataIngestionConfig,
                                        DataTransformationConfig,
                                        ModelTrainingConfig,
                                        ModelEvaluationConfig,
                                        FeatureImportanceConfig,
                                        TablesConfig
                                        )
def get_feature_importance_config(self) -> FeatureImportanceConfig:
        config = self.config.feature_importance 
        schema = self.schema.target_column
        params= self.params
        
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

# create configuration manager 
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
        
        
    def get_data_generation_config(self)-> DataGenerationConfig:
        """
        This method is responsible for creating the data generation configuration"""
    
        config = self.config.data_generation
        
        
        create_directories([self.config.artifacts_root])
        
        data_generation_config = DataGenerationConfig(
            num_customers = config.num_customers,
            output_dir = Path(config.output_dir),
            customers_filename = config.customers_filename,
            products_filename = config.products_filename,
            gen_root_dir = Path(config.gen_root_dir),
            data_dir  = Path(config.data_dir),
            db_file = Path(config.db_file),
            table = config.table        )
        
        return data_generation_config
    
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        table_config = self.config.data_generation.table
    
        create_directories([self.config.artifacts_root])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            raw_data_dir=Path(config.raw_data_dir),
            customers_table=table_config.customers,
            products_table=table_config.products,
            data_file=Path(config.data_file),
            customers_csv=Path(config.customers_csv),
            products_csv=Path(config.products_csv)
            
        )
        
        return data_ingestion_config
    
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
            product_path= Path(config.product_path),
            customer_path= Path(config.customer_path),
            train_data_file= Path(config.train_data_file),
            test_data_file= Path(config.test_data_file),
            target_column= schema.name,
            joined_data_file= Path(config.joined_data_file)
            )
        
        
        return data_transformation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params.random_forest
        schema =  self.schema.target_column

        create_directories([self.config.artifacts_root])

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            train_data_dir = Path(config.train_data_dir),
            test_data_dir = Path(config.test_data_dir),
            model_name = config.model_name,
            criterion = params.criterion,
            max_features = params.max_features,
            min_samples_split = params.min_samples_split,
            min_samples_leaf = params.min_samples_leaf,
            n_estimators = params.n_estimators, 
            max_depth = params.max_depth,
            random_state = params.random_state,
            class_weight = params.class_weight,
            n_jobs = params.n_jobs,
            target_column = schema.name
            
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
            grid_search_model_path= Path(config.grid_search_model_path)
            
            
           
            
        )

        return model_evaluation_config
    
    def get_feature_importance_config(self) -> FeatureImportanceConfig:
        config = self.config.feature_importance 
        schema = self.schema.target_column
        params= self.params
        
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
    
    
    