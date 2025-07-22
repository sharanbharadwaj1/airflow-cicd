import pandas as pd
import joblib
from google.cloud import storage
from google.cloud import bigquery
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import json
import logging
import gcsfs
from imblearn.over_sampling import RandomOverSampler # Correct import
import io # Added for in-memory serialization

# Configure basic logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Encodes categorical columns using one-hot encoding.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_cols (list): A list of column names to be one-hot encoded.

    Returns:
        pd.DataFrame: The DataFrame with specified categorical columns encoded.
    """
    logger.info(f"Encoding categorical columns: {categorical_cols}")
    # Use pd.get_dummies for one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def apply_bucketing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies bucketing to the 'age' column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'age' column bucketed.
    """
    logger.info("Applying bucketing to 'age' column.")
    # Define age bins and labels
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    df['age_bucket'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    # One-hot encode the new age_bucket column
    df = pd.get_dummies(df, columns=['age_bucket'], drop_first=True)
    df = df.drop('age', axis=1) # Drop original age column
    return df

def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses features by separating X and y, and handling any final cleaning.

    Args:
        df (pd.DataFrame): The input DataFrame after encoding and bucketing.

    Returns:
        tuple[pd.DataFrame, pd.Series]: X (features DataFrame) and y (target Series).
    """
    logger.info("Preprocessing features: separating X and y, and final cleaning.")
    # Assuming 'y' is the target variable
    if 'y' not in df.columns:
        raise ValueError("Target column 'y' not found in the DataFrame.")
    
    X = df.drop('y', axis=1)
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0) # Convert 'yes'/'no' to 1/0
    
    # Ensure all column names are string type (important for XGBoost)
    X.columns = X.columns.astype(str)

    return X, y

def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains an XGBoost classification model.

    Args:
        model_name (str): The name of the model (expected to be "xgboost").
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        XGBClassifier: The trained XGBoost model.
    """
    logger.info(f"Training {model_name} model.")
    if model_name == "xgboost":
        # Initialize and train XGBoost classifier
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
        model.fit(X_train, y_train)
        return model
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

def get_classification_report(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Generates a classification report for the trained model.

    Args:
        model: The trained model (e.g., XGBClassifier).
        X (pd.DataFrame): Features for evaluation.
        y (pd.Series): True labels for evaluation.

    Returns:
        dict: A dictionary containing the classification report metrics.
    """
    logger.info("Generating classification report.")
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    return report

def save_model_artifact(model_name: str, model_pipeline, bucket_name: str = 'sharan-mlops'):
    """
    Saves the trained model artifact to Google Cloud Storage.

    Args:
        model_name (str): Name of the model (e.g., "xgboost").
        model_pipeline: The trained model or pipeline object.
        bucket_name (str): The name of the GCS bucket to save the model to.
    """
    logger.info(f"Saving {model_name} model artifact to GCS bucket: {bucket_name}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    blob_name = f"bank_campaign_model_artifacts/{model_name}_model_{timestamp}.joblib"
    
    # Use BytesIO to serialize the model to an in-memory byte stream
    buffer = io.BytesIO()
    joblib.dump(model_pipeline, buffer)
    model_bytes = buffer.getvalue() # Get the bytes from the buffer

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Upload the byte stream
    blob.upload_from_string(model_bytes, content_type='application/octet-stream')
    logger.info(f"Model artifact saved to gs://{bucket_name}/{blob_name}")

def write_metrics_to_bigquery(algo_name: str, training_time: datetime, metrics: dict, 
                               dataset_id: str = 'mlops', table_id: str = 'bank_campaign_model_metrics',
                               project_id: str = 'mlops-ld'):
    """
    Writes model evaluation metrics to a BigQuery table.

    Args:
        algo_name (str): Name of the algorithm used (e.g., "xgboost").
        training_time (datetime): Timestamp of the model training.
        metrics (dict): Dictionary of evaluation metrics.
        dataset_id (str): BigQuery dataset ID.
        table_id (str): BigQuery table ID.
        project_id (str): Google Cloud Project ID.
    """
    logger.info(f"Writing metrics to BigQuery table: {project_id}.{dataset_id}.{table_id}")
    client = bigquery.Client(project=project_id)
    table_ref = client.dataset(dataset_id).table(table_id)

    # Ensure the table exists and has the correct schema
    # For simplicity, we assume the table and schema are already created.
    # Schema: algo_name (STRING), training_time (TIMESTAMP), model_metrics (STRING - JSON)
    
    row_to_insert = [{
        "algo_name": algo_name,
        "training_time": training_time.isoformat(), # BigQuery expects ISO format for TIMESTAMP
        "model_metrics": json.dumps(metrics) # Store metrics as a JSON string
    }]

    errors = client.insert_rows_json(table_ref, row_to_insert)

    if errors:
        logger.error(f"Errors while inserting rows into BigQuery: {errors}")
        raise RuntimeError(f"Failed to insert metrics into BigQuery: {errors}")
    else:
        logger.info("Metrics successfully written to BigQuery.")

def main():
    """
    Main function to orchestrate the data loading, preprocessing, model training,
    evaluation, and saving of artifacts and metrics.
    """
    input_data_path = "gs://sharan-mlops/bank-additional-full.csv"
    model_name = "xgboost"

    logger.info(f"Starting main execution for model training with data from: {input_data_path}")

    # Load data
    fs = gcsfs.GCSFileSystem()
    with fs.open(input_data_path) as f:
        df = pd.read_csv(f, sep=";")
    logger.info("Data loaded successfully.")

    # Define categorical columns
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    # Preprocess data
    df_encoded = encode_categorical(df.copy(), categorical_cols)
    df_bucketed = apply_bucketing(df_encoded.copy())
    X, y = preprocess_features(df_bucketed.copy())
    logger.info("Data preprocessing complete.")

    # Apply oversampling
    oversampler = RandomOverSampler(random_state=42) # Corrected from RandomSampler to RandomOverSampler
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    logger.info("Oversampling applied.")

    # Train model
    pipeline = train_model(model_name, X_resampled, y_resampled)
    logger.info("Model training complete.")

    # Get model metrics
    model_metrics = get_classification_report(pipeline, X_resampled, y_resampled)
    logger.info(f"Model metrics: {model_metrics}")

    # Save model artifact
    save_model_artifact(model_name, pipeline)
    logger.info("Model artifact saved.")

    # Write metrics to BigQuery
    write_metrics_to_bigquery(model_name, datetime.now(), model_metrics)
    logger.info("Metrics written to BigQuery.")

if __name__ == "__main__":
    main()
