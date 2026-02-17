#!/usr/bin/env python3
"""
verify_setup.py - Verify that the scoring pipeline produces expected outputs.

Run this script to confirm your environment is correctly configured.

Usage:
    python verify_setup.py           # Standard output
    python verify_setup.py --debug   # Include intermediate values for troubleshooting
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
import time, os
import warnings

# Import preprocessing from local copy
from preprocessing import (
    FEEDBACK_IDENTITY_FEATURES,
    FEEDBACK_IDENTITY_BASE_FEATURES,
    IDENTITY_ONLY_FEATURES,
    IDENTITY_ONLY_BASE_FEATURES,
    compute_derived_features_feedback_identity,
    compute_derived_features_identity_only,
)

warnings.filterwarnings('ignore')
TOLERANCE = 0.1  # Acceptable difference for authorshipLikelihoodScore (0-100 scale)

def read_json_file(file_name, timeout=60):
    """
    Reads a JSON file from local disk with optional wait for the file to appear.

    Args:
        file_name (str): Name of the file to read
        timeout (int): Maximum seconds to wait for the file

    Returns:
        dict: Parsed JSON content

    Raises:
        FileNotFoundError: If file does not appear within the timeout
        json.JSONDecodeError: If the file is not valid JSON
        Exception: For any other errors during file read
    """
    file_path = os.path.join("/var/task/data", file_name)
    logging.info(f"Preparing to read file: {file_path}")

    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            logging.error(f"File '{file_path}' not found after {timeout} seconds")
            raise FileNotFoundError(f"{file_path} not found after {timeout} seconds")
        logging.info(f"Waiting for file '{file_path}' to appear...")
        time.sleep(1)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Successfully read file '{file_path}'")
            return data
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file '{file_path}': {e}")
        raise
    except Exception as e:
        logging.exception(f"Unexpected error reading file '{file_path}'")
        raise


def read_file_from_s3(bucket_name, file_name):
    """
    Reads a JSON file from S3 and returns it as a Python dictionary.

    Args:
        bucket_name (str): Name of the S3 bucket
        file_name (str): Name of the file in the S3 bucket

    Returns:
        dict: Parsed JSON content

    Raises:
        FileNotFoundError: If the file does not exist in S3
        json.JSONDecodeError: If the file content is not valid JSON
        Exception: For other unexpected errors
    """
    s3 = boto3.client('s3')
    logging.info(f"Attempting to read S3 file '{file_name}' from bucket '{bucket_name}'")

    try:
        # Attempt to get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        content = response['Body'].read().decode('utf-8')  # Decode bytes to string
        logging.info(f"Successfully retrieved file '{file_name}' from S3")
        
        # Parse JSON content
        data = json.loads(content)
        return data

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logging.error(f"File '{file_name}' does not exist in bucket '{bucket_name}'")
            raise FileNotFoundError(f"S3 file '{file_name}' not found in bucket '{bucket_name}'")
        else:
            logging.exception(f"A client error occurred while accessing S3: {e}")
            raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file '{file_name}' from bucket '{bucket_name}': {e}")
        raise
    except Exception as e:
        logging.exception(f"Unexpected error reading file '{file_name}' from S3")
        raise
    finally:
        logging.info(f"Finished attempting to read file '{file_name}' from S3")


def file_exists_in_s3(bucket_name, file_name):
    """
    Checks if a file exists in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket
        file_name (str): Name of the file/key to check

    Returns:
        bool: True if the file exists, False if it does not exist

    Raises:
        Exception: For unexpected errors accessing S3
    """
    s3 = boto3.client('s3')
    logging.info(f"Checking existence of S3 file '{file_name}' in bucket '{bucket_name}'")

    try:
        # Try to retrieve metadata for the object
        s3.head_object(Bucket=bucket_name, Key=file_name)
        logging.info(f"File '{file_name}' exists in bucket '{bucket_name}'")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404' or error_code == 'NoSuchKey':
            logging.warning(f"File '{file_name}' does not exist in bucket '{bucket_name}'")
            return False
        else:
            logging.exception(f"ClientError when checking file '{file_name}' in bucket '{bucket_name}': {e}")
            raise
    except Exception as e:
        logging.exception(f"Unexpected error when checking file '{file_name}' in bucket '{bucket_name}': {e}")
        raise


def main():
    
    try:
        try:
            from urllib3.exceptions import SNIMissingWarning
        except ImportError:
            # Handle the absence or use an alternative
            SNIMissingWarning = None

        # Ignore SNIMissingWarning
        warnings.filterwarnings("ignore", category=UserWarning, message=".*SNI.*")
    
        # Set up logging configuration
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler(sys.stderr))

        startTime = time.perf_counter();
    
        logging.info(f'executing main method {startTime}')

        parser = argparse.ArgumentParser(description="Verify CART scoring pipeline")
        parser.add_argument(
            "--debug", action="store_true",
                                
            help="Show intermediate values (raw_probability, calibrated_probability) for troubleshooting"
        )
        # Set up argument parsing
        parser.add_argument('modelName', type=str, help='The name of the string tells which model to load')
        parser.add_argument('file_name', type=str, help='The name of the JSON file to read')
        parser.add_argument('bucket_name', type=str, help='The name of the S3 bucket')
        parser.add_argument('useS3Bucket', type=str, help='Flag whether to use S3 Bucket or not')

        args = parser.parse_args()
        logging.info(f'args {args}');
        
        s3 = boto3.client('s3')
                    

        logging.info(f"=" * 70)
        logging.info("CART Scoring Pipeline Verification")
        logging.info(f"=" * 70)   

        base_dir = Path(__file__).parent
        errors = []
        logging.info(f'base_dir: {base_dir}');       
                                                                        
        # 1. Load models
        logging.info("\n1. Loading models...")
        try:
            if args.modelName == "feedback":
                fb_model = joblib.load(base_dir / "feedbackIdentityModel.joblib")
                fb_cal = joblib.load(base_dir / "feedbackIdentityCalibrator.joblib")
                fb_scaler = joblib.load(base_dir / "feedbackIdentityScaler.joblib")
            else :
                io_model = joblib.load(base_dir / "identityOnlyModel.joblib")
                io_cal = joblib.load(base_dir / "identityOnlyCalibrator.joblib")
                io_scaler = joblib.load(base_dir / "identityOnlyScaler.joblib")
                logging.info("OK - All models loaded")
        except Exception as e:
            logging.exception(f"Error loading models: {e}")
            return {"status": "error", "message": f"Model loading failed: {e}"}
        
        # 2. Load Data
        try:
            logging.info(f"The bucket flag '{args.useS3Bucket}' exists in the bucket '{args.bucket_name}'.")
            if args.useS3Bucket == "false":
                logging.info('reading the file from File folder:')
                articles = read_json_file(args.file_name)
                df = pd.DataFrame(articles)
            elif args.useS3Bucket == "true" and file_exists_in_s3(args.bucket_name, args.file_name):
                logging.info(f"The file '{args.file_name}' exists in the bucket '{args.bucket_name}'.")
                # Proceed to read the file from S3
                articles = read_file_from_s3(args.bucket_name, args.file_name)
            else:
                logging.warning(f"Invalid useS3Bucket flag value: {args.useS3Bucket}")
                return {"status": "error", "message": f"Invalid useS3Bucket flag: {args.useS3Bucket}"}

            if not articles:
                logging.warning(f"No data loaded from file '{args.file_name}'")
                return {"status": "error", "message": "No data loaded from input file"}
			
            df = pd.DataFrame(articles)
        except Exception as e:
            logging.exception(f"Error loading input data: {e}")
            return {"status": "error", "message": f"Data loading failed: {e}"}
        
            # 3. Preprocessing
        try:
            if args.modelName == "feedback" :
                logging.info("\n4. Preprocessing for Feedback+Identity model...")
                df_fb = df.copy()
                for feat in FEEDBACK_IDENTITY_BASE_FEATURES:
                    if feat not in df_fb.columns:
                        df_fb[feat] = 0
                    df_fb[feat] = df_fb[feat].fillna(0)
                #logging.info("feedback columns scores*****\n%s", df_fb)    
                df_fb = compute_derived_features_feedback_identity(df_fb)
                #logging.info("compute_derived_features_feedback*****\n%s", df_fb)  
                logging.info(f"   OK - {len(FEEDBACK_IDENTITY_FEATURES)} features prepared")

            elif args.modelName =="identity" :
                logging.info("\n5. Preprocessing for Identity-Only model...")
                df_io = df.copy()
                for feat in IDENTITY_ONLY_BASE_FEATURES:
                    if feat not in df_io.columns:
                        df_io[feat] = 0
                    df_io[feat] = df_io[feat].fillna(0)
                df_io = compute_derived_features_identity_only(df_io)
                logging.info(f"   OK - {len(IDENTITY_ONLY_FEATURES)} features prepared")
            else:
                logging.warning(f"Invalid modelName: {args.modelName}")
                return {"status": "error", "message": f"Invalid modelName: {args.modelName}"}
        
        except Exception as e:
            logging.exception(f"Error during preprocessing: {e}")
            return {"status": "error", "message": f"Preprocessing failed: {e}"}
            
            # 4. Scoring
        try:    
            logging.info("\n6. Scoring articles...")
            if args.modelName =="feedback" :
                logging.info("feedback_identity_features:\n%s",df_fb[FEEDBACK_IDENTITY_FEATURES].to_string())
                X_fb = fb_scaler.transform(df_fb[FEEDBACK_IDENTITY_FEATURES].values)
                #logging.info("X_fb\n%s:", X_fb)
                raw_fb = fb_model.predict_proba(X_fb)[:, 1]
                #logging.info("raw fb\n%s:", raw_fb)
                cal_fb = fb_cal.predict(raw_fb.reshape(-1, 1))
                #logging.info("cal fb\n%s:", cal_fb)
                score_fb = cal_fb * 100
                #logging.info("score_fb:\n%s",score_fb)
                # Prepare the output
                # Make a dictionary for each row
                scoring_output = [
                    {'id': article_id, 'scoreTotal': score}
                    for article_id, score in zip(df_fb['articleId'], score_fb)
                ]
                logging.info("scoring_output:\n%s",scoring_output)
                print(json.dumps(scoring_output, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))

            elif args.modelName == "identity":    
                X_io = io_scaler.transform(df_io[IDENTITY_ONLY_FEATURES].values)
                raw_io = io_model.predict_proba(X_io)[:, 1]
                cal_io = io_cal.predict(raw_io)
                score_io = cal_io * 100
                logging.info(f'score_io {cal_io}')

                # Prepare the output
                # Make a dictionary for each row
                scoring_output = [
                    {'id': article_id, 'scoreTotal': score}
                    for article_id, score in zip(df_io['articleId'], score_io)
                ]
                logging.info("scoring_output:\n%s",scoring_output)
                logging.info("   OK - Predictions generated")

        except Exception as e:
            logging.exception(f"Error during scoring: {e}")
            return {"status": "error", "message": f"Scoring failed: {e}"}
    
    except Exception as e:
        logging.exception(f"Unexpected error in main(): {e}")
        return {"status": "error", "message": f"Unexpected failure: {e}"}


if __name__ == "__main__":
    try:
        main()  # Call the main function
    except Exception as e:
        import traceback
        import json
        import logging

        # Log the full stack trace
        logging.error("Unhandled exception in main():\n%s", traceback.format_exc())

        # Optionally, return the error to a calling program or API
        error_output = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_output))  # So a calling program can parse it

