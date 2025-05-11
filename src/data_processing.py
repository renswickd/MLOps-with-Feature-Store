import pandas as pd
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, train_data_path, test_data_path, feature_store: RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data = None
        self.test_data = None
        self.feature_store = feature_store

        logger.info("Data Processing class has been initialized...")
        logger.info(f"Train data path: {self.train_data_path}")
        logger.info(f"Test data path: {self.test_data_path}")

    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Data has been loaded successfully...")
            logger.info(f"Train data shape: {self.data.shape}")
            logger.info(f"Test data shape: {self.test_data.shape}")
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise CustomException("Data loading failed", str(e))

    def preprocess_data(self):
        try:
            # Identify categorical and numerical columns
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            numerical_columns = self.data.select_dtypes(exclude=['object']).columns

            # Handle missing values
            self.data[numerical_columns] = self.data[numerical_columns].fillna(self.data[numerical_columns].mean())
            self.test_data[numerical_columns] = self.test_data[numerical_columns].fillna(self.test_data[numerical_columns].mean())

            # Apply one-hot encoding for categorical columns
            self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)
            self.test_data = pd.get_dummies(self.test_data, columns=categorical_columns, drop_first=True)

            # Align train and test datasets to have the same columns
            self.test_data = self.test_data.reindex(columns=self.data.columns, fill_value=0)

            logger.info("Data preprocessing has been done successfully...")
            logger.info(f"Train data shape after preprocessing: {self.data.shape}")
            logger.info(f"Test data shape after preprocessing: {self.test_data.shape}")
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise CustomException("Data preprocessing failed: ", str(e))

    def store_feature_in_redis(self):
        try:
            batch_data = {}
            for idx, row in self.data.iterrows():
                entity_id = f"train_{idx}"
                features = row.to_dict()
                batch_data[entity_id] = features

            for idx, row in self.test_data.iterrows():
                entity_id = f"test_{idx}"
                features = row.to_dict()
                batch_data[entity_id] = features

            self.feature_store.store_batch_features(batch_data)
            logger.info("Data has been stored in Redis successfully...")
        except Exception as e:
            logger.error(f"Data storing into Redis failed: {e}")
            raise CustomException("Data storing into Redis failed: ", str(e))

    def run(self):
        try:
            logger.info("Data Processing Pipeline Begins...")
            self.load_data()
            self.preprocess_data()
            self.store_feature_in_redis()
            logger.info("Data Processing Pipeline Completed Successfully...")
        except Exception as e:
            logger.error(f"Data Processing Pipeline failed: {e}")
            raise CustomException("Data Processing Pipeline failed: ", str(e))

if __name__ == "__main__":
    feature_store = RedisFeatureStore()

    data_processor = DataProcessing(TRAIN_PATH, TEST_PATH, feature_store)
    data_processor.run()