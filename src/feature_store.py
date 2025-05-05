import redis
import json
import logging
from src.custom_exception import CustomException

# Configure logging
logger = logging.getLogger(__name__)

class RedisFeatureStore:
    def __init__(self, host="localhost", port=6379, db=0):
        try:
            self.client = redis.StrictRedis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            # Test REDIS connection
            self.client.ping()
            logger.info("Connected to Redis successfully.")
        except redis.ConnectionError as e:
            logger.error("Failed to connect to Redis: %s", e)
            raise CustomException("Could not connect to Redis.", e)

    # Store one row
    def store_features(self, entity_id, features):
        try:
            key = f"entity:{entity_id}:features"
            self.client.set(key, json.dumps(features))
            logger.info("Stored features for entity_id: %s", entity_id)
        except Exception as e:
            logger.error("Failed to store features for entity_id %s: %s", entity_id, e)
            raise CustomException(f"Failed to store features for entity_id {entity_id}.", e)

    # get one row
    def get_features(self, entity_id):
        try:
            key = f"entity:{entity_id}:features"
            features = self.client.get(key)
            if features:
                logger.info("Retrieved features for entity_id: %s", entity_id)
                return json.loads(features)
            logger.warning("No features found for entity_id: %s", entity_id)
            return None
        except Exception as e:
            logger.error("Failed to get features for entity_id %s: %s", entity_id, e)
            raise CustomException(f"Failed to get features for entity_id {entity_id}.", e)

    # Storing batch data
    def store_batch_features(self, batch_data):
        try:
            for entity_id, features in batch_data.items():
                self.store_features(entity_id, features)
            logger.info("Stored batch features for %d entities.", len(batch_data))
        except Exception as e:
            logger.error("Failed to store batch features: %s", e)
            raise CustomException("Failed to store batch features.", e)

    # Getting batch data
    def get_batch_features(self, entity_ids):
        try:
            batch_features = {}
            for entity_id in entity_ids:
                batch_features[entity_id] = self.get_features(entity_id)
            logger.info("Retrieved batch features for %d entities.", len(entity_ids))
            return batch_features
        except Exception as e:
            logger.error("Failed to get batch features: %s", e)
            raise CustomException("Failed to get batch features.", e)

    # Getting all entity IDs
    def get_all_entity_ids(self):
        try:
            keys = self.client.keys('entity:*:features')
            entity_ids = [key.split(':')[1] for key in keys]
            logger.info("Retrieved all entity IDs. Total: %d", len(entity_ids))
            return entity_ids
        except Exception as e:
            logger.error("Failed to get all entity IDs: %s", e)
            raise CustomException("Failed to get all entity IDs.", e)