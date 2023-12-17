import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split


# Strategy Interface
class DataStrategy(ABC):
    """Abstract class defining strategy for handling data"""

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass



# Concrete Strategy
class DataPreProcessStrategy(DataStrategy):
    """Strategy for preprocessing data"""

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data"""
        try:
            # columns deleted for simplicity but they do have importance
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ], 
                axis=1,
            )

            # Handle null values in data frame by replacing with 
            # the median value of that column
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

            # Replace null value with "No Review"
            data["review_comment_message"].fillna("No Review", inplace=True)

            # Select only numeric data types
            data = data.select_dtypes(include=np.number)
            cols_to_drop = ["custopmer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))



# Concrete Strategy
class DataDivideStrategy(DataStrategy):
    """Strategy for dividing data into train and test set"""
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """Divide data into train and test"""
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # X_train, X_test => pd.DataFrame
            # y_train, y_test => pd.Series
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e



# Context: this defines the interface for the clients
class DataCleaning:
    """Class that will preprocess and make the train test split"""
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e



# Client/application that makes use of the strategy pattern
# if __name__ == "__main__":
#     data = pd.read_csv("/Users/louisevelayo/Documents/GitHub/node-monitor-new/customer-satisfaction-mlops/data/olist_customers_dataset.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaning.handle_data()


            