
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from BankProducts import logger
from dotenv import load_dotenv
import os
from sqlalchemy import text 
from BankProducts.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
        self.engine = self._create_engine_from_env()

    def _create_engine_from_env(self):
        """Load DB credentials and return SQLAlchemy engine."""
        load_dotenv()
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

        return create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

    def initiate_data_ingestion(self):
        """Load local CSV into PostgreSQL database."""
        logger.info("Starting data ingestion process...")

        # Check if file exists
        csv_path = self.config.local_data_file
        if not os.path.exists(csv_path):
            logger.error("CSV file not found. Please check the file path.")
            return

        try:
            df = pd.read_csv(csv_path)
            df.to_sql("bank_transactions", self.engine, if_exists="replace", index=False)
            logger.info("Dataset successfully loaded into PostgreSQL.")
        except Exception as e:
            logger.error(f"Failed to load data into PostgreSQL: {e}")

    def extract_and_save_data(self):
        """Fetch data from DB and save to CSV."""
        logger.info("Extracting and saving data...")

        query = "SELECT * FROM bank_transactions;"

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT to_regclass('bank_transactions')"))
                table_exists = result.scalar() is not None

            if not table_exists:
                logger.error("Table 'bank_transactions' does not exist in the database.")
                return

            logger.info("Fetching data from the 'bank_transactions' table...")
            df = pd.read_sql(query, self.engine)
            output_path = self.config.output_path
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)   
            df.to_csv(self.config.output_path, index=False)
            logger.info(f"Data successfully exported to '{output_path}'.")

        except Exception as e:
            logger.error(f"Error while extracting data from database: {e}")