import pandas as pd
import sqlite3
from pathlib import Path
from BankProducts import  logger
from BankProducts.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        logger.info("Data Validation started")

    def validate_all_columns(self) -> bool:
        try:
            data = pd.read_csv(self.config.customer_data)
            logger.info(f"Data loaded from {self.config.customer_data}")

            # Normalize column names to lowercase
            data.columns = data.columns.str.lower()
            schema_cols = {col.lower(): dtype for col, dtype in self.config.all_schema.items()}

            # Optional: Convert datetime columns
            if 'transactiondate' in data.columns:
                data['transactiondate'] = pd.to_datetime(data['transactiondate'])

            if data.empty:
                logger.warning("Data is empty")
                raise ValueError("Customer data is empty.")

            logger.info(f"Data shape: {data.shape}")

            data_cols = set(data.columns)
            schema_keys = set(schema_cols.keys())

            missing_in_data = schema_keys - data_cols
            extra_in_data = data_cols - schema_keys

            validation_status = not missing_in_data and not extra_in_data
            
            os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}\n")
                if missing_in_data:
                    f.write(f"Missing columns: {missing_in_data}\n")
                if extra_in_data:
                    f.write(f"Extra columns: {extra_in_data}\n")

            logger.info(f"Validation completed. Status: {validation_status}")
            return validation_status

        except Exception as e:
            logger.exception("Exception occurred during data validation")
            raise e
