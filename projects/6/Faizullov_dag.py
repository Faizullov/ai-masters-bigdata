import json
import os

from textwrap import dedent
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

import pendulum

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

#AIRFLOW_HOME=os.environ.get("AIRFLOW_HOME", 
#                            f'{os.environ["HOME"]}/airflow')

TRAIN_PATH="/datasets/amazon/all_reviews_5_core_train_extra_small_sentiment.json"
TRAIN_PATH_OUT="Faizullov_hw6_train_out"

TEST_PATH="/datasets/amazon/all_reviews_5_core_test_extra_small_features.json"
TEST_PATH_OUT="Faizullov_hw6_test_out"

#TRUE_PATH="$dirname/true_labels/pre16extra_small_labels.csv"
PRED_PATH="Faizullov_hw6_prediction"

dsenv="/opt/conda/envs/dsenv/bin/python"


with DAG(
    'Faizullov_dag',
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={'retries': 2},
    description='ETL DAG hw6',
    schedule_interval=None,
    start_date=pendulum.datetime(2023, 1, 3, tz="UTC"),
    catchup=False
) as dag:
    
    base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'

    feature_eng_train_task = SparkSubmitOperator(
          application=f"{base_dir}/feature_eng_task.py",
          application_args=["--path-in", TRAIN_PATH, "--path-out", TRAIN_PATH_OUT],
          task_id="feature_eng_train_task",
          spark_binary="/usr/bin/spark3-submit",
        )

    download_train_task = BashOperator(
         task_id='download_train_task',
         bash_command = f'pwd; echo \$USER; hdfs dfs -getmerge {TRAIN_PATH_OUT} {base_dir}/{TRAIN_PATH_OUT}_local'
    )


    from pathlib import Path

    #TODO workdir?
    train_task = BashOperator(
         task_id='train_task',
         bash_command=f'{dsenv} {base_dir}/train.py --train-in {base_dir}/{TRAIN_PATH_OUT}_local --sklearn-model-out {base_dir}/6.joblib'
    )

    #
    # Test
    #
    
    # Model Sensor
    model_sensor = FileSensor(
      task_id=f'model_sensor',
      filepath=f"{base_dir}/6.joblib",
      poke_interval=20,
      timeout=20 * 20,
      #mode="reschedule",
      #soft_fail=False
    )

    # Featur eng
    feature_eng_test_task = SparkSubmitOperator(
          application=f"{base_dir}/feature_eng_test_task.py",
          application_args=["--path-in", TEST_PATH, "--path-out", TEST_PATH_OUT],
          task_id="feature_eng_test_task",
          spark_binary="/usr/bin/spark3-submit",
    )

    predict_task = SparkSubmitOperator(
          application=f"{base_dir}/predict.py",
          application_args=["--test-in", TEST_PATH_OUT, "--pred-out", PRED_PATH, "--sklearn-model-in", f"6.joblib"],
          task_id="predict_task",
          spark_binary="/usr/bin/spark3-submit",
          files=f'{base_dir}/6.joblib',
          env_vars={"PYSPARK_PYTHON": dsenv}
    )
 

    feature_eng_train_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> predict_task