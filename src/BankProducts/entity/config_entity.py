from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    #export_csv_path: Path
    output_path: Path
    table_name: str
    
@dataclass(frozen=True)
class DataValidationConfig:
    val_root_dir: Path
    STATUS_FILE: str
    customer_data: Path
    all_schema: dict
    customer_data: Path
    

@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Data Transformation Configuration
    """
    root_dir: Path
    transformed_data_file: Path
    customer_path: Path
    train_data_file: Path
    test_data_file: Path
    target_column: str

@dataclass(frozen=True)
class ModelTrainingConfig:
    """Configuration for model training.
    """
    model_name: str
    root_dir: Path
    test_data_dir: Path
    train_data_dir: Path
    random_state: int
    class_weight: str
    n_jobs: int
    target_column: str
    label_encoder_file: str
    C: float
    penalty: str
    solver: str
    max_iter: int
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column: str
    params: dict[str, str]
    grid_search_model_path: Path
    train_data_path: Path
    encoded_target_label: Path
    

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
    
@dataclass(frozen=True)
class FinalModelConfig: 
    """Configuration for the final model"""
    
    training_data: Path
    testing_data: Path
    final_model: Path
    target_column: str
    penalty: str
    C: float
    solver: str
    max_iter: int
    random_state: int
    class_weight: str
    n_jobs: int
    model_encoder: Path
    metric_file: Path