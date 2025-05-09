from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
from src.feature_store import RedisFeatureStore
from config.hyperparameters import parameter_lists
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
from sklearn.metrics import accuracy_score

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self , feature_store:RedisFeatureStore , model_save_path = "artifacts/models/"):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None

        os.makedirs(self.model_save_path , exist_ok=True)
        logger.info("Model Training Begins....")

    def load_data_from_redis(self , entity_ids):
        try:
            logger.info("Data Extraction from REDIS begins....")

            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning("No Features found for entity_id: %s", entity_id)
            logger.info("Data Extraction from REDIS completed....")
            return data
        except Exception as e:
            logger.error(f"Error while loading data from REDIS {e}")
            raise CustomException(str(e))
        
    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()

            train_entity_ids , test_entity_ids = train_test_split(entity_ids , test_size=0.2 , random_state=42)

            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_train, X_test = train_df.drop('Survived',axis=1), test_df.drop('Survived',axis=1)
            y_train, y_test = train_df["Survived"], test_df["Survived"]
            logger.info(f"Train data shape : {X_train.shape}\n Test data shape : {X_test.shape}")
            return X_train , X_test , y_train, y_test
        
        except Exception as e:
            logger.error(f"Data preparation failed {e}")
            raise CustomException(str(e))
        
    def hyperparamter_tuning(self,X_train,y_train):
        try:
            
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(rf, parameter_lists, n_iter=10, cv=3, scoring='accuracy', random_state=42)
            random_search.fit(X_train, y_train)

            logger.info(f"Best paramters : {random_search.best_params_}")
            return random_search.best_estimator_
        
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed {e}")
            raise CustomException(str(e))
        
    def train_and_evaluate(self , X_train , y_train , X_test , y_test):
        try:
            best_rf = self.hyperparamter_tuning(X_train,y_train)
            y_pred = best_rf.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            logger.info(f"Accuracy is {accuracy}")
            self.save_model(best_rf)

            return accuracy
        
        except Exception as e:
            logger.error(f"Error while model training {e}")
            raise CustomException(str(e))
    
    def save_model(self , model):
        try:
            model_filename = f"{self.model_save_path}rf_model.pkl"

            with open(model_filename,'wb') as model_file:
                pickle.dump(model , model_file)

            logger.info(f"Model saved at {model_filename}")
        except Exception as e:
            logger.error(f"Error while model saving {e}")
            raise CustomException(str(e))
        
    def run(self):
        try:
            logger.info("Starting Model Training Pipleine....")
            X_train , X_test , y_train, y_test = self.prepare_data()
            accuracy = self.train_and_evaluate(X_train , y_train, X_test , y_test)
            logger.info(f"Model training completed with accuracy: {accuracy}")
            logger.info("End of Model Training pipeline...")

        except Exception as e:
            logger.error(f"Model training failed {e}")
            raise CustomException(str(e))
        
if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = ModelTraining(feature_store)
    model_trainer.run()