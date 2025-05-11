# from src.logger import get_logger
# from src.custom_exception import CustomException
# import pandas as pd
# from src.feature_store import RedisFeatureStore
# from config.hyperparameters import parameter_lists
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier
# import os
# import pickle
# from sklearn.metrics import accuracy_score

# logger = get_logger(__name__)

# class ModelTraining:

#     def __init__(self , feature_store:RedisFeatureStore , model_save_path = "artifacts/models/"):
#         self.feature_store = feature_store
#         self.model_save_path = model_save_path
#         self.model = None

#         os.makedirs(self.model_save_path , exist_ok=True)
#         logger.info("Model Training Begins....")

#     def load_data_from_redis(self , entity_ids):
#         try:
#             logger.info("Data Extraction from REDIS begins....")

#             data = []
#             for entity_id in entity_ids:
#                 features = self.feature_store.get_features(entity_id)
#                 if features:
#                     data.append(features)
#                 else:
#                     logger.warning("No Features found for entity_id: %s", entity_id)
#             logger.info("Data Extraction from REDIS completed....")
#             return data
#         except Exception as e:
#             logger.error(f"Error while loading data from REDIS {e}")
#             raise CustomException(str(e))
        
#     def prepare_data(self):
#         try:
#             entity_ids = self.feature_store.get_all_entity_ids()

#             train_entity_ids , test_entity_ids = train_test_split(entity_ids , test_size=0.2 , random_state=42)

#             train_data = self.load_data_from_redis(train_entity_ids)
#             test_data = self.load_data_from_redis(test_entity_ids)

#             train_df = pd.DataFrame(train_data)
#             test_df = pd.DataFrame(test_data)

#             X_train, X_test = train_df.drop('Survived',axis=1), test_df.drop('Survived',axis=1)
#             y_train, y_test = train_df["Survived"], test_df["Survived"]
#             logger.info(f"Train data shape : {X_train.shape}\n Test data shape : {X_test.shape}")
#             return X_train , X_test , y_train, y_test
        
#         except Exception as e:
#             logger.error(f"Data preparation failed {e}")
#             raise CustomException(str(e))
        
#     def hyperparamter_tuning(self,X_train,y_train):
#         try:
            
#             rf = RandomForestClassifier(random_state=42)
#             random_search = RandomizedSearchCV(rf, parameter_lists, n_iter=10, cv=3, scoring='accuracy', random_state=42)
#             random_search.fit(X_train, y_train)

#             logger.info(f"Best paramters : {random_search.best_params_}")
#             return random_search.best_estimator_
        
#         except Exception as e:
#             logger.error(f"Hyperparameter tuning failed {e}")
#             raise CustomException(str(e))
        
#     def train_and_evaluate(self , X_train , y_train , X_test , y_test):
#         try:
#             best_rf = self.hyperparamter_tuning(X_train,y_train)
#             y_pred = best_rf.predict(X_test)
#             accuracy = accuracy_score(y_test,y_pred)
#             logger.info(f"Accuracy is {accuracy}")
#             self.save_model(best_rf)

#             return accuracy
        
#         except Exception as e:
#             logger.error(f"Error while model training {e}")
#             raise CustomException(str(e))
    
#     def save_model(self , model):
#         try:
#             model_filename = f"{self.model_save_path}rf_model.pkl"

#             with open(model_filename,'wb') as model_file:
#                 pickle.dump(model , model_file)

#             logger.info(f"Model saved at {model_filename}")
#         except Exception as e:
#             logger.error(f"Error while model saving {e}")
#             raise CustomException(str(e))
        
#     def run(self):
#         try:
#             logger.info("Starting Model Training Pipleine....")
#             X_train , X_test , y_train, y_test = self.prepare_data()
#             accuracy = self.train_and_evaluate(X_train , y_train, X_test , y_test)
#             logger.info(f"Model training completed with accuracy: {accuracy}")
#             logger.info("End of Model Training pipeline...")

#         except Exception as e:
#             logger.error(f"Model training failed {e}")
#             raise CustomException(str(e))
        
# if __name__ == "__main__":
#     feature_store = RedisFeatureStore()
#     model_trainer = ModelTraining(feature_store)
#     model_trainer.run()

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, feature_store: RedisFeatureStore, model_save_path="artifacts/model/random_forest_model.pkl"):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.data = None
        self.test_data = None

        logger.info("ModelTraining class has been initialized...")

    def load_data_from_redis(self):
        try:
            # Retrieve all entity IDs from Redis
            entity_ids = self.feature_store.get_all_entity_ids()

            # Separate train and test entity IDs
            train_ids = [eid for eid in entity_ids if eid.startswith("train_")]
            test_ids = [eid for eid in entity_ids if eid.startswith("test_")]

            # Retrieve features for train and test datasets
            train_data = self.feature_store.get_batch_features(train_ids)
            test_data = self.feature_store.get_batch_features(test_ids)

            # Convert to pandas DataFrame
            self.data = pd.DataFrame.from_dict(train_data, orient="index")
            self.test_data = pd.DataFrame.from_dict(test_data, orient="index")

            logger.info("Data has been loaded from Redis successfully...")
            logger.info(f"Train data shape: {self.data.shape}")
            logger.info(f"Test data shape: {self.test_data.shape}")
        except Exception as e:
            logger.error(f"Failed to load data from Redis: {e}")
            raise CustomException("Failed to load data from Redis", str(e))

    def train_model(self):
        try:
            # print(self.data.columns)
            self.data.columns = [i.strip() for i in self.data.columns]
            # Define predictor (target) and features
            X = self.data.drop(columns=["Life expectancy"])
            y = self.data["Life expectancy"]

            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize the Random Forest Regressor
            rf_regressor = RandomForestRegressor(random_state=42, n_estimators=100)

            # Train the model
            rf_regressor.fit(X_train, y_train)

            # Validate the model
            y_pred = rf_regressor.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            logger.info(f"Validation Mean Squared Error: {mse}")
            logger.info(f"Validation R^2 Score: {r2}")

            return rf_regressor
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise CustomException("Model training failed", str(e))

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            joblib.dump(model, self.model_save_path)
            logger.info(f"Model saved to {self.model_save_path}")
        except Exception as e:
            logger.error(f"Failed to save the model: {e}")
            raise CustomException("Failed to save the model", str(e))

    def run(self):
        try:
            logger.info("Model Training Pipeline Begins...")
            self.load_data_from_redis()
            model = self.train_model()
            self.save_model(model)
            logger.info("Model Training Pipeline Completed Successfully...")
        except Exception as e:
            logger.error(f"Model Training Pipeline failed: {e}")
            raise CustomException("Model Training Pipeline failed", str(e))

if __name__ == "__main__":
    feature_store = RedisFeatureStore()

    model_trainer = ModelTraining(feature_store)
    model_trainer.run()