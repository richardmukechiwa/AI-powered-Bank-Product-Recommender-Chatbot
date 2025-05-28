from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TablesConfig:
    customers: str
    products: str

@dataclass(frozen=True)
class DataGenerationConfig:
    num_customers: int
    output_dir: Path
    customers_filename: str
    products_filename: str
    gen_root_dir: Path
    data_dir: Path
    db_file: Path
    table: TablesConfig
    
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    raw_data_dir: Path
    data_file: Path
    customers_csv: Path
    products_csv: Path
    customers_table: str
    products_table: str
    

@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Data Transformation Configuration
    """
    root_dir: Path
    transformed_data_file: Path
    customer_path: Path
    product_path: Path
    train_data_file: Path
    test_data_file: Path
    target_column: str
    joined_data_file: Path
    
    
@dataclass(frozen=True)
class ModelTrainingConfig:
    """Configuration for model training.
    """
    model_name: str
    root_dir: Path
    test_data_dir: Path
    train_data_dir: Path
    criterion: str
    max_features: int
    min_samples_split: int
    min_samples_leaf: int
    n_estimators: int
    max_depth: int
    random_state: int
    class_weight: str
    n_jobs: int
    target_column: str
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column: str
    params: dict[str, str]
    grid_search_model_path: Path
    
@dataclass(frozen=True)
class FeatureImportanceConfig:
    """Configuration for feature importance analysis.
    """
    root_dir: Path
    grid_search_model: Path
    training_data_path: Path
    test_data_path: Path
    feature_importance_file: Path
    target_column: str
    