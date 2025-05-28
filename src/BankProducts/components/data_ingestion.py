import os
import logging
import sqlite3
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from BankProducts import logger
from BankProducts.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_and_save_data(self):
        """Extract customers and products data from SQLite DB and save as CSV."""
        logging.info(f"Connecting to database: {self.config.data_file}")
        conn = sqlite3.connect(self.config.data_file)

        try:
            customers_df = pd.read_sql_query(f"SELECT * FROM {self.config.customers_table}", conn)
            products_df = pd.read_sql_query(f"SELECT * FROM {self.config.products_table}", conn)
            logging.info(f"Successfully read tables {self.config.customers_table} and {self.config.products_table}")
        finally:
            conn.close()
            logging.info("Database connection closed.")

        # Ensure directories exist
        Path(self.config.raw_data_dir).mkdir(parents=True, exist_ok=True)

        # Save to CSV at correct file paths
        customers_df.to_csv(self.config.customers_csv, index=False)
        products_df.to_csv(self.config.products_csv, index=False)
        logging.info(f"Customers data saved to: {self.config.customers_csv}")
        logging.info(f"Products data saved to: {self.config.products_csv}")

        return customers_df, products_df