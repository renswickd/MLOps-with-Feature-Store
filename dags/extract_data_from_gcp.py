from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base_hook import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy

#### TRANSFORM STEP....
def upload_to_sql(file_path):
    conn = BaseHook.get_connection('postgres_default')  
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@mlops-with-redis_ca7fc5-postgres-1:{conn.port}/{conn.schema}")
    df = pd.read_csv(file_path)
    df.to_sql(name="titanic", con=engine, if_exists="replace", index=False)

# Define the DAG
with DAG(
    dag_id="data_pipeline_gcs_to_sql",
    schedule_interval=None, 
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    list_files = GCSListObjectsOperator(
        task_id="extract_all_files",
        bucket="e2e-mlops-with-redis", 
    )

    download_file = GCSToLocalFilesystemOperator(
        task_id="extract_file",
        bucket="e2e-mlops-with-redis", 
        object_name="Titanic-Dataset.csv", 
        filename="/tmp/Titanic-Dataset.csv", 
    )
    
    load_data = PythonOperator(
        task_id="load_data_to_sql",
        python_callable=upload_to_sql,
        op_kwargs={"file_path": "/tmp/Titanic-Dataset.csv"}
    )

    list_files >> download_file >> load_data
