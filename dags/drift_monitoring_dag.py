from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'raj',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def run_drift_detector():
    os.system("python3 detector/drift_detector.py")

def run_retrain_model():
    os.system("python3 retraining/retrain_model.py")

with DAG(
    dag_id='model_drift_monitoring_dag',
    default_args=default_args,
    start_date=datetime(2025, 4, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['drift', 'retraining'],
) as dag:

    detect_drift = PythonOperator(
        task_id='detect_drift',
        python_callable=run_drift_detector
    )

    retrain_model = PythonOperator(
        task_id='retrain_model',
        python_callable=run_retrain_model
    )

    detect_drift >> retrain_model
