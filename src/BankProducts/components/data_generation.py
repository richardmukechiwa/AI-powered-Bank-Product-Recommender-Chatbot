import pandas as pd
import sqlite3
from pathlib import Path
from BankProducts import logger
from faker import Faker
import pandas as pd
import random
from sqlalchemy import create_engine
from pathlib import Path
from BankProducts.entity.config_entity import DataGenerationConfig
import os


fake =Faker()

# Define product catalog
PRODUCT_CATALOG = [
    {
        "product_name": "Savings Account",
        "description": "A basic savings account with competitive interest rates.",
        "eligibility": "All customers above 18 years old"
    },
    {
        "product_name": "Credit Card",
        "description": "A credit card with cashback and reward points.",
        "eligibility": "Credit score above 650 and income above $20,000"
    },
    {
        "product_name": "Home Loan",
        "description": "Flexible home loan with low interest rates.",
        "eligibility": "Credit score above 700 and income above $50,000"
    },
    {
        "product_name": "Education Loan",
        "description": "Loan for students pursuing higher education.",
        "eligibility": "Age below 35 and enrollment in a valid institution"
    },
    {
        "product_name": "Fixed Deposit",
        "description": "Investment with fixed returns over a chosen term.",
        "eligibility": "Minimum deposit of $1,000"
    }
    ]

class DataGeneration:
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        

    def generate_customer_data(self, num_records=15000):
        data = []
        products = [p["product_name"] for p in PRODUCT_CATALOG]
        goals = ["Home Ownership", "Education", "Savings", "Travel", "Retirement"]
        
        logger.info(f"Generating {num_records} fake customer records...")
        for _ in range(num_records):
            data.append({
                "customer_id": fake.uuid4(),
                "name": fake.name(),
                "age": random.randint(18, 70),
                "gender": random.choice(["Male", "Female"]),
                "occupation": fake.job(),
                "annual_income": round(random.uniform(15000, 200000), 2),
                "marital_status": random.choice(["Single", "Married", "Divorced"]),
                "credit_score": random.randint(300, 850),
                "existing_products": ', '.join(random.sample(products, k=random.randint(0, 3))),
                "financial_goals": random.choice(goals)
            })
        
        customers_df = pd.DataFrame(data)
        products_df = pd.DataFrame(PRODUCT_CATALOG)
        
        logger.info("Customer and product data generated.")
        return customers_df, products_df

    def save_to_csv(self, customers_df, products_df, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        customers_path = output_dir / "bank_customers.csv"
        products_path = output_dir / "product_catalog.csv"
        
        customers_df.to_csv(customers_path, index=False)
        products_df.to_csv(products_path, index=False)
        
        logger.info(f"Saved customers to {customers_path}")
        logger.info(f"Saved products to {products_path}")
        
        return customers_path, products_path

    def save_to_db(self, customers_path: str, products_path: str, db_file):
        try:
            # Check if files exist
            if not os.path.exists(customers_path):
                logger.error(f"Customer file not found: {customers_path}")
                raise FileNotFoundError(f"Customer file not found: {customers_path}")
            if not os.path.exists(products_path):
                logger.error(f"Product file not found: {products_path}")
                raise FileNotFoundError(f"Product file not found: {products_path}")
            
            # Load CSV files
            customers = pd.read_csv(customers_path)
            products = pd.read_csv(products_path)

            # Create SQLite engine
            engine = create_engine(f"sqlite:///{db_file}")

            # Write to database
            customers.to_sql("customers", con=engine, if_exists="replace", index=False)
            products.to_sql("products", con=engine, if_exists="replace", index=False)

            logger.info(f"Data saved to SQLite database at {db_file}")
            return db_file

        except Exception as e:
            logger.exception(f"Failed to save data to the database: {e}")
            raise





