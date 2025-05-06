# MLOps with Feature Store

## Project Overview

This project demonstrates an end-to-end MLOps pipeline with a focus on integrating a feature store for efficient feature management. The system is designed to handle the entire machine learning lifecycle, including data ingestion, transformation, feature storage, model training, and deployment. 

The feature store is implemented using Redis, which serves as a high-performance in-memory database for storing and retrieving features.

### Key Components
1. **ETL Pipelines**: Extract, Transform, and Load pipelines for data preprocessing and feature engineering, implemented using Astro Airflow.
2. **Feature Store**: A centralized repository for storing and retrieving features, implemented using Redis.
3. **Model Training**: Automated pipelines for training machine learning models using the features stored in the feature store.
4. **Deployment**: Deployment of trained models for inference.

---

## Setup Instructions

### Prerequisites
- Docker installed and running.
- Python 3.8+ installed.
- Redis installed locally or running in a Docker container.
- Astro CLI installed (see [Astro CLI Installation Guide](https://docs.astronomer.io/astro/cli/install-cli)).
- Required Python dependencies installed (see `requirements.txt`).

---

### 1. Setting Up the Redis Feature Store

#### Using Docker
1. Pull the Redis image:
   ```bash
   docker pull redis
   ```
2. Run the Redis container:
   ```bash
   docker run -d --name redis-feature-store -p 6379:6379 redis
   ```
3. Verify that Redis is running:
   ```bash
   docker ps
   ```

---

### 2. Setting Up the ETL Pipelines with Astro Airflow

The ETL pipelines are implemented using Astro Airflow to orchestrate the data ingestion, transformation, and feature storage processes.

#### Steps:
1. **Initialize an Astro Project**:
   - If you haven’t already, initialize an Astro project:
     ```bash
     astro dev init
     ```
   - This will create a project directory with the necessary files for running Airflow locally.

2. **Configure the DAG**:
   - Add your ETL pipeline logic to a DAG file in the `dags/` directory (e.g., `etl_pipeline_dag.py`).
   - Ensure the DAG includes tasks for:
     - Extracting raw data.
     - Transforming the data into features.
     - Storing the features in the Redis feature store.

3. **Update the `Dockerfile`**:
   - Ensure the `Dockerfile` includes all necessary dependencies for your ETL pipeline, such as Redis and Python libraries:
     ```dockerfile
     FROM quay.io/astronomer/astro-runtime:7.3.0
     RUN pip install redis
     ```

4. **Start the Astro Airflow Environment**:
   - Start the local Astro Airflow environment:
     ```bash
     astro dev start
     ```
   - This will spin up the Airflow webserver, scheduler, and other necessary components.

5. **Access the Airflow UI**:
   - Open the Airflow UI in your browser at `http://localhost:8080`.
   - Use the default credentials:
     - Username: `admin`
     - Password: `admin`

6. **Trigger the ETL Pipeline**:
   - In the Airflow UI, locate your DAG (e.g., `etl_pipeline_dag`) and trigger it manually or set a schedule for automatic execution.

7. **Verify Feature Storage**:
   - After the DAG runs successfully, verify that the features have been stored in Redis:
     ```bash
     redis-cli
     keys *
     ```

---

### 3. Integrating the Feature Store

The feature store is implemented in `src/feature_store.py`. It provides methods for storing and retrieving features.

#### Key Methods:
- `store_features(entity_id, features)`: Stores features for a specific entity.
- `get_features(entity_id)`: Retrieves features for a specific entity.
- `store_batch_features(batch_data)`: Stores features for multiple entities in a batch.
- `get_batch_features(entity_ids)`: Retrieves features for multiple entities in a batch.

#### Example Usage:
```python
from src.feature_store import RedisFeatureStore

# Initialize the feature store
feature_store = RedisFeatureStore(host="localhost", port=6379)

# Store features for an entity
feature_store.store_features("entity_1", {"feature1": 0.5, "feature2": 1.2})

# Retrieve features for an entity
features = feature_store.get_features("entity_1")
print(features)
```

---

## Additional Notes

- **Error Handling**: Custom exceptions are implemented in `src/custom_exception.py` to handle errors gracefully.
- **Logging**: Detailed logging is implemented across all scripts for easier debugging.
- **Scalability**: The feature store can be scaled by deploying Redis in a clustered mode.
- **Airflow Best Practices**: Follow Astro Airflow best practices for DAG design and task orchestration.

For further details, refer to the individual scripts in the `src` directory.