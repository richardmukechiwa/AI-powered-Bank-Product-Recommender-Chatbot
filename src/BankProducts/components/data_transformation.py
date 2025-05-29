import os  
from BankProducts import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from BankProducts.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        #self.data = None
        #self.transformed_data = None
        
    def join_datasets(self):
        """
        Join customer and product datasets"""    
        try:
            customer_data = pd.read_csv(self.config.customer_path)
            product_data = pd.read_csv(self.config.product_path)

            # Ensure the directory for saving exists
            output_dir = os.path.dirname(self.config.transformed_data_file)
            os.makedirs(output_dir, exist_ok=True)

            # Join operation
            joined_data = pd.merge(customer_data, product_data, how="left",
                                left_on="existing_products", right_on="product_name")
            
            #drop unnecessary columns
            joined_data = joined_data.drop(columns=['existing_products'], errors='ignore')  # Drop columns which are not needed if it exists
            
            #check the dataset head
            print(joined_data.head())
            
            
            # Save the joined data
            try:   
                joined_data.to_csv(self.config.joined_data_file, index=False)
                logger.info(f"Joined dataset saved to {self.config.joined_data_file}")
                print(f"Joined dataset saved to {self.config.joined_data_file}")
            except Exception as e:
                logger.error(f"Error saving joined dataset: {e}")
                print(f"Error saving joined dataset: {e}")
            
            return joined_data

        except Exception as e:
            logger.error(f"Error in joining datasets: {e}")
            print(f"Error in joining datasets: {e}")
            raise e
    def transform_data(self):
        """
        Transform the data as per the requirements
        """
        
        # Load the data
        data = pd.read_csv(self.config.joined_data_file)
        # Perform transformations
        print(data.head())
        
        print(":"*100)
        
        data.info()
        print(":"*100)
        
        data.describe()
        print(":"*100)
        print(data.columns)
        print(":"*100)
        
        #drop na
        data.dropna(inplace=True
                            )
        #check null values
        print(data.isnull().sum())
        
        #check the number of  values in the target column
        print(data[self.config.target_column].value_counts())
        
        #resize the dataset to match the number of rows in the target column
        data = data[data[self.config.target_column].notnull()]
        
        #drop unnecessary columns
        data = data.drop(columns=['customer_id','name', 'eligibility', 'description'], errors='ignore')
        
        # print the first 5 rows of the data
        data.head()
        
        
        
        #plot "product_name" histogram based of gender using seaborn
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(data=data, x= self.config.target_column, hue='gender', multiple='stack')

        # Add separated count labels above each segment
        for container in ax.containers:
            # Add offset so overlapping labels are vertically separated
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,  # Place label at the middle of the segment
                        f'{int(height)}',
                        ha='center',
                        va='center',
                        fontsize=9,
                        color='white',  # or 'black' depending on your bar color
                        weight='bold'
                    )


        plt.title("Product Name Histogram by Gender")
        plt.tight_layout()
        plt.show()
                    
        #plot "age" histogram
        plt.figure(figsize=(10,6))
        plt.hist(data["age"], bins=10, edgecolor='black', color= "orange", alpha=0.7)
        plt.title("Age Frequency Distribution")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.savefig("age_histogram.png")
        
    
        
        #plot "product_name" vs "age" bar plot
        plt.figure(figsize=(10,6))
        plt.bar(data["product_name"], data["age"], color="green")
        plt.title("Product Name vs Age Bar Plot")
        plt.xlabel("Product Name")
        plt.ylabel("Age")
        plt.savefig("product_name_vs_age_bar_plot.png")
        
        #feature selection
        # If it's a categorical variable like a string, correlation won't work correctly
        correlation_matrix = data.select_dtypes(include= ['float64', 'int64']).corr()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
        plt.title("Correlation Matrix")
        plt.show()
        plt.savefig('correlation_matrix.png')
    
        # Save the data to a CSV file to the specified path
        os.makedirs(os.path.dirname(self.config.transformed_data_file), exist_ok=True)
        data.to_csv(self.config.transformed_data_file, index=False)
        logger.info(f"Transformed data saved to {self.config.transformed_data_file}")
        print(f"Transformed data saved to {self.config.transformed_data_file}")
        
        return data
      
    def split_data(self):
        data =  pd.read_csv(self.config.transformed_data_file)  
        
        #  splitting data into train and test sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        # print the first 5 rows of the train and test data
        print("Train Data:")
        print(train_data.head())
        
        print("Test Data:") 
        print(test_data.head())
        
        
        #save train_data and test_data to csv files
        train_data.to_csv(os.path.join(self.config.train_data_file), index=False)
        test_data.to_csv(os.path.join(self.config.test_data_file), index=False)
        
        logger.info(f"Train and test sets saved to {self.config.train_data_file} and {self.config.test_data_file}")
        print(f"Train and test sets saved to {self.config.train_data_file} and {self.config.test_data_file}")
        # Log the shapes of the train and test sets
        logger.info(f"Train set shape: {train_data.shape}, Test set shape: {test_data.shape}")
        #print the shapes of the train and test sets
        print(f"Train set shape: {train_data.shape}, Test set shape: {test_data.shape}")
        # Log the shapes of the train and test sets
        return train_data, test_data    
        
        
      

