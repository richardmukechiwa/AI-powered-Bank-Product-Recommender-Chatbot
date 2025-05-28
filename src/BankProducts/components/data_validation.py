import pandas as pd
import sqlite3
from pathlib import Path
from BankProducts import  logger
from BankProducts.entity.config_entity import DataGenerationConfig

class DataValidation:
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        
    def validate_file_exists(self, path: Path, name: str):
        path = Path(path)
        if not path.is_absolute():
            path = self.config.gen_root_dir / path
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at: {path}")
        print(f" {name} exists at {path}")

    def validate_csv_not_empty(self, path: Path, name: str):
        path= Path(path)
        if not path.is_absolute():
            path = self.config.gen_root_dir / path
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"{name} is empty.")
        print(f" {name} is not empty with {len(df)} rows")

    def validate_database_tables(self):
        expected = self.config.table
        with sqlite3.connect(self.config.db_file) as conn:
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            actual_tables = {row[0] for row in result.fetchall()}

        missing = []
        for table in [expected.customers, expected.products]:
            if table not in actual_tables:
                missing.append(table)

        if missing:
            raise ValueError(f"Missing tables: {missing}")
        print(f" All expected tables exist in the DB: {expected.customers}, {expected.products}")
        logger.info(f"Data validation completed successfully.")
