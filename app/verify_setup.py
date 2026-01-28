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
from botocore.exceptions import NoCredentialsError
import time, os

# Import preprocessing from local copy
from preprocessing import (
    FEEDBACK_IDENTITY_FEATURES,
    FEEDBACK_IDENTITY_BASE_FEATURES,
    IDENTITY_ONLY_FEATURES,
    IDENTITY_ONLY_BASE_FEATURES,
    compute_derived_features_feedback_identity,
    compute_derived_features_identity_only,
)


import warnings
warnings.filterwarnings('ignore')

TOLERANCE = 0.1  # Acceptable difference for authorshipLikelihoodScore (0-100 scale)


def read_json_file(file_name):
    try:
        logging.info(f"Filename for input processing. {file_name}")
        file_path = os.path.join("/var/task/data", file_name)
        logging.info(f'file_path {file_path}')
        while not os.path.exists(file_path):
            time.sleep(1)
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data  # Return the loaded JSON data
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_name}' was not found.")
    except json.JSONDecodeError:
        logging.error("Error: The file is not a valid JSON.")
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")
        sys.exit(1)  # Exit if there's an error in loading the data

def read_file_from_s3(bucket_name, file_name):
    s3 = boto3.client('s3')

    try:
        # Get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        
        # Read the content of the file
        content = response['Body'].read().decode('utf-8')  # Decode bytes to string
        
        # If the file is JSON, load it into a Python dictionary
        data = json.loads(content)
        return data

    except Exception as e:
        logging.error(f"Error reading file from S3: {e}")
        return None
    
    finally:
        logging.info("Finished reading the JSON file from S3")
        # Call the upload function with your S3 bucket name
        #upload_log_to_s3()

def file_exists_in_s3(bucket_name, file_name):
    s3 = boto3.client('s3')
    logging.info(f"Checking S3 Bucket: {bucket_name}, File: {file_name}")  # Log bucket and file

    try:
        # Try to retrieve metadata for the object
        s3.head_object(Bucket=bucket_name, Key=file_name)
        return True  # If no exception, the file exists
    except ClientError as e: 
        # If a 404 error is raised, the file does not exist
        if e.response['Error']['Code'] == '404':
            return False
        else:
            # Raise other errors
            raise

def main():
    
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
    logging.info(f"   FAIL - {e}")
    sys.exit(1)

logging.info(f"The bucket flag '{args.useS3Bucket}' exists in the bucket '{args.bucket_name}'.")
if args.useS3Bucket == "false":
    logging.info('reading the file from File folder:')
    articles = read_json_file(args.file_name)
    df = pd.DataFrame(articles)
if args.useS3Bucket == "true" and file_exists_in_s3(args.bucket_name, args.file_name):
    logging.info(f"The file '{args.file_name}' exists in the bucket '{args.bucket_name}'.")
    # Proceed to read the file from S3
    articles = read_file_from_s3(args.bucket_name, args.file_name)
    df = pd.DataFrame(articles)
    if articles is not None:
        logging.info(articles)
    else:
        logging.info(f"Error: The file '{args.file_name}' does not exist locally and the file '{args.file_name}' does not exist in the bucket '{args.bucket_name}' and useS3Bucket is '{args.useS3Bucket}'.")
	
if args.modelName == "feedback" :
    logging.info("\n4. Preprocessing for Feedback+Identity model...")
    df_fb = df.copy()
    for feat in FEEDBACK_IDENTITY_BASE_FEATURES:
        if feat not in df_fb.columns:
            df_fb[feat] = 0
        df_fb[feat] = df_fb[feat].fillna(0)
    logging.info("feedback columns scores*****\n%s", df_fb)    
    df_fb = compute_derived_features_feedback_identity(df_fb)
    logging.info("compute_derived_features_feedback*****\n%s", df_fb)  
    logging.info(f"   OK - {len(FEEDBACK_IDENTITY_FEATURES)} features prepared")

    # 5. Preprocess for Identity-Only
if args.modelName =="identity" :
    logging.info("\n5. Preprocessing for Identity-Only model...")
    df_io = df.copy()
    for feat in IDENTITY_ONLY_BASE_FEATURES:
        if feat not in df_io.columns:
            df_io[feat] = 0
        df_io[feat] = df_io[feat].fillna(0)
    df_io = compute_derived_features_identity_only(df_io)
    logging.info(f"   OK - {len(IDENTITY_ONLY_FEATURES)} features prepared")

# 6. Scale and predict
logging.info("\n6. Scoring articles...")
if args.modelName =="feedback" :
    logging.info("feedback_identity_features:\n%s",df_fb[FEEDBACK_IDENTITY_FEATURES].to_string())
    X_fb = fb_scaler.transform(df_fb[FEEDBACK_IDENTITY_FEATURES].values)
    logging.info("X_fb\n%s:", X_fb)
    raw_fb = fb_model.predict_proba(X_fb)[:, 1]
    logging.info("raw fb\n%s:", raw_fb)
    cal_fb = fb_cal.predict(raw_fb.reshape(-1, 1))
    logging.info("cal fb\n%s:", cal_fb)
    score_fb = cal_fb * 100
    logging.info("score_fb:\n%s",score_fb)
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

# Final output: authorshipLikelihoodScore = calibrated × 100
logging.info("   OK - Predictions generated")


if __name__ == "__main__":
    main()
