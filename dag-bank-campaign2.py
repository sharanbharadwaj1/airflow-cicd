# !pip3 install xgboost (This is a comment for local testing, packages should be in requirements.txt for Composer)

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
from google.cloud import storage
import gcsfs,json
from datetime import datetime
from google.cloud import bigquery
from google.cloud import logging as cloud_logging # ALIASED: Import Google Cloud Logging as cloud_logging
import logging # Python's standard logging module
from imblearn.over_sampling import RandomOverSampler

# Import functions from your local training script
from bank_campaign_model_training import (
    encode_categorical, 
    apply_bucketing, 
    preprocess_features,
    write_metrics_to_bigquery, 
    get_classification_report, 
    save_model_artifact,
    train_model
)

# Initialize Python's standard logger for general messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set logging level

# Initialize Google Cloud Logging client for structured logs
# Now using the aliased 'cloud_logging'
cloud_logging_client = cloud_logging.Client() 
cloud_logger = cloud_logging_client.logger('bank-campaign-training-logs')


def validate_csv():
    """
    Validates the input CSV data by checking for expected columns.
    Logs an error and raises a ValueError if columns do not match.
    """
    logger.info("Starting CSV validation.") # Using standard logger
    # Load data from GCS
    fs = gcsfs.GCSFileSystem()
    # Ensure this GCS path is correct and accessible by Composer's service account
    with fs.open("gs://sharan-mlops/bank-additional-full.csv") as f:
        df = pd.read_csv(f, sep=";")
    
    # Define expected columns for validation
    expected_cols = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 
                     'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
                     'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                     'euribor3m', 'nr.employed', 'y']
    
    # Check if the loaded columns are same as expected columns
    if list(df.columns) == expected_cols:
        logger.info("CSV columns validated successfully.") # Using standard logger
        return True
    else:
        # Log structured error message to Cloud Logging
        cloud_logger.log_struct({ # Using cloud_logger for structured logs
            'keyword': 'Bank_Campaign_Model_Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Input Data is not valid",
            'training_status':0
        })
        raise ValueError(f'CSV does not have expected columns. Columns in CSV are: {list(df.columns)}')

def read_last_training_metrics():
    """
    Reads the latest model metrics for 'xgboost' from the BigQuery table.
    """
    logger.info("Reading last training metrics from BigQuery.") # Using standard logger
    # Initialize BigQuery client, explicitly setting the project for robustness
    # Ensure 'mlops-ld' is your correct project ID
    client = bigquery.Client(project='mlops-ld') 
    
    # Fully qualify the table ID with project and dataset
    # This is crucial for BigQuery to find the table correctly
    table_id = "mlops-ld.mlops.bank_campaign_model_metrics" 
    
    query = f"""
        SELECT model_metrics
        FROM `{table_id}`
        WHERE algo_name='xgboost'
        ORDER BY training_time DESC
        LIMIT 1
    """
    
    result = client.query(query).result()
    
    # Check if any rows were returned
    first_row = next(result, None) # Use next with a default to avoid StopIteration
    
    if first_row:
        # Access by index (0 for model_metrics as per query)
        metrics_json_string = first_row[0] 
        logger.info("Last training metrics read successfully.") # Using standard logger
        return json.loads(metrics_json_string)
    else:
        logger.warning("No previous 'xgboost' model metrics found in BigQuery. Assuming initial state.") # Using standard logger
        # Return default metrics if no previous data exists,
        # to prevent errors in evaluate_model's comparison.
        # This structure should match what get_classification_report returns for '0' class.
        return {'0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}}


def evaluate_model():
    """
    Evaluates the model, compares its metrics with the last saved model,
    and saves the new model artifact and metrics if thresholds are met.
    """
    logger.info("Starting model evaluation.") # Using standard logger
    # Load data for evaluation
    fs = gcsfs.GCSFileSystem()
    with fs.open('gs://sharan-mlops/bank-additional-full.csv') as f:
        df = pd.read_csv(f, sep=";")
    
    # Preprocess the data using functions from bank_campaign_model_training
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                        'day_of_week', 'poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)

    # Apply oversampling
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Train the model on new data 
    model_name = "xgboost"
    pipeline = train_model(model_name, X_resampled, y_resampled)

    # Get the current model metrics for evaluation
    model_metrics = get_classification_report(pipeline, X_resampled, y_resampled)
    
    # Safely get precision and recall for class '0'
    # Ensure '0' key exists in model_metrics before accessing
    precision = model_metrics.get('0', {}).get('precision', 0.0)
    recall = model_metrics.get('0', {}).get('recall', 0.0)

    # Get the last/existing model metrics for comparison
    last_model_metrics = read_last_training_metrics()
    last_precision = last_model_metrics.get('0', {}).get('precision', 0.0)
    last_recall = last_model_metrics.get('0', {}).get('recall', 0.0)

    # Define the threshold values for precision and recall
    precision_threshold = 0.98
    recall_threshold = 0.98
    
    # Decision logic: save if current metrics meet thresholds
    if (precision >= precision_threshold and recall >= recall_threshold):
        save_model_artifact(model_name, pipeline)
        write_metrics_to_bigquery("xgboost", datetime.now(), model_metrics)
        cloud_logger.log_struct({ # Using cloud_logger for structured logs
            'keyword': 'Bank_Campaign_Model_Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Model artifact saved",
            'training_status':1,
            'current_metrics': model_metrics,
            'last_metrics': last_model_metrics
        })
        logger.info("Model artifact saved and metrics written to BigQuery as thresholds met.") # Using standard logger
    else:
        cloud_logger.log_struct({ # Using cloud_logger for structured logs
            'keyword': 'Bank_Campaign_Model_Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Model metrics do not meet the defined threshold",
            'model_metrics': model_metrics,
            'last_model_metrics': last_model_metrics, 
            'precision_threshold': precision_threshold,
            'recall_threshold': recall_threshold,
            'training_status':0
        })
        logger.warning("Model metrics did not meet the defined thresholds. Model not saved.") # Using standard logger


# Define the default_args dictionary for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1), 
    'retries': 1,              
}

# Instantiate the DAG
dag = DAG(
    'dag_bank_campaign_continuous_training',
    default_args=default_args,
    description='A not so simple training DAG for bank campaign model continuous training',
    schedule_interval=None, 
    catchup=False,          
    tags=['mlops', 'bank_campaign', 'training'] 
)

# Define the tasks/operators
validate_csv_task = PythonOperator(
    task_id='validate_csv',
    python_callable=validate_csv,
    dag=dag,
)

evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=evaluate_model,
    dag=dag,
)

# Define task dependencies
validate_csv_task >> evaluation_task
