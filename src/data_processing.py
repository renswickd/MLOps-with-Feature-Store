import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, train_data_path , test_data_path , feature_store : RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data=None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train=None
        self.y_test = None

        self.X_resampled = None
        self.y_resampled = None

        self.feature_store = feature_store
        logger.info("Data Processing class has been initialized...")
        logger.info(f"Train data path : {self.train_data_path}")
        logger.info(f"Test data path : {self.test_data_path}")
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Data has been loaded successfully...")
            logger.info(f"Train data shape : {self.data.shape}")
            logger.info(f"Test data shape : {self.test_data.shape}")
        except Exception as e:
            logger.error(f"Data loading Failed: {e}")
            raise CustomException(str(e))
    
    def preprocess_data(self):
        try:
            self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
            self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
            self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())
            self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})
            self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes


            self.data['Familysize'] = self.data['SibSp'] + self.data['Parch'] + 1
            self.data['Isalone'] = (self.data['Familysize'] == 1).astype(int)
            self.data['HasCabin'] = self.data['Cabin'].notnull().astype(int)
            self.data['Title'] = self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}).fillna(4)
            self.data['Pclass_Fare'] = self.data['Pclass'] * self.data['Fare']
            self.data['Age_Fare'] = self.data['Age'] * self.data['Fare']

            logger.info("data preprocessing has been done successfully...")
            logger.info(f"Data shape after preprocessing : {self.data.shape}")

        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise CustomException(str(e))
    
    def handle_imbalance_data(self):
        try:
            X = self.data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']]
            y = self.data['Survived']

            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(X, y)

            logger.info("Class imbalance handling has been done successfully...")
            logger.info(f"Data shape after handling imbalance : {self.X_resampled.shape}")
            logger.info(f"Data shape after handling imbalance : {self.y_resampled.shape}")

        except Exception as e:
            logger.error(f"Imbalance handling failed: {e}")
            raise CustomException(str(e))
    
    def store_feature_in_redis(self):
        try:
            batch_data = {}
            for idx,row in self.data.iterrows():
                entity_id = row["PassengerId"]
                features = {
                    "Age" : row['Age'],
                    "Fare" : row["Fare"],
                    "Pclass" : row["Pclass"],
                    "Sex" : row["Sex"],
                    "Embarked" : row["Embarked"],
                    "Familysize": row["Familysize"],
                    "Isalone" : row["Isalone"],
                    "HasCabin" : row["HasCabin"],
                    "Title" : row["Title"],
                    "Pclass_Fare" : row["Pclass_Fare"],
                    "Age_Fare" : row["Age_Fare"],
                    "Survived" : row["Survived"]
                }
                batch_data[entity_id] = features
            self.feature_store.store_batch_features(batch_data)
            logger.info("data has been stored in redis successfully...")
        except Exception as e:
            logger.error(f"Data storing into REDIS failed: {e}")
            raise CustomException(str(e))
        
    def run(self):
        try:
            logger.info("Data Processing Pipeline Begins...")
            self.load_data()
            self.preprocess_data()
            self.handle_imbalance_data()
            self.store_feature_in_redis()

            logger.info("Data Processing Pipeline Completed Successfully...")

        except Exception as e:
            logger.error(f"Data Processing Pipeline failed: {e}")
            raise CustomException(str(e))
        
if __name__=="__main__":
    feature_store = RedisFeatureStore()

    data_processor = DataProcessing(TRAIN_PATH,TEST_PATH,feature_store)
    data_processor.run()
        


