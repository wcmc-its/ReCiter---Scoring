#!/usr/bin/env python3
"""
verify_setup.py - Verify that the scoring pipeline produces expected outputs.

Run this script to confirm your environment is correctly configured.

Usage:
    python verify_setup.py           # Standard output
    python verify_setup.py --debug   # Include intermediate values for troubleshooting
"""

import json
import sys
from pathlib import Path

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

def read_json_file(scoringDataFile, timeout=60):
    """
    Reads a JSON file from local disk with optional wait for the file to appear.

    Args:
        scoringDataFile (str): Name of the file to read
        timeout (int): Maximum seconds to wait for the file

    Returns:
        dict: Parsed JSON content

    Raises:
        FileNotFoundError: If file does not appear within the timeout
        json.JSONDecodeError: If the file is not valid JSON
        Exception: For any other errors during file read
    """
    file_path = os.path.join("/var/task/data", scoringDataFile)
    logging.info(f"Preparing to read file: {file_path}")

    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            logging.exception(f"Unexpected error reading file '{scoringDataFile}' from S3")
            return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"{file_path} not found after {timeout} seconds"
                }
        logging.info(f"Waiting for file '{file_path}' to appear...")
        time.sleep(1)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Successfully read file '{file_path}'")
            return data
    except FileNotFoundError:
        logging.exception(f"Unexpected error reading file '{scoringDataFile}' from S3")
        return {
                "predictionScores": "",
                "returnCode": 1,
                "error": f"Unexpected error reading file '{scoringDataFile}' from S3"
            }
    except json.JSONDecodeError as e:
        logging.exception(f"Unexpected error reading file '{scoringDataFile}' from S3")
        return {
                "predictionScores": "",
                "returnCode": 1,
                "error": f"Unexpected error reading file '{scoringDataFile}' from S3"
            }
    except Exception as e:
        logging.exception(f"Unexpected error reading files '{file_path}' and '{scoringDataFile}' from S3")
        return {
            "predictionScores": "",
            "returnCode": 1,
            "error": f"Unexpected error reading file '{scoringDataFile}' from S3"
        }


def read_file_from_s3(bucket_name, scoringDataFile):
    """
    Reads a JSON file from S3 and returns it as a Python dictionary.

    Args:
        bucket_name (str): Name of the S3 bucket
        scoringDataFile (str): Name of the file in the S3 bucket

    Returns:
        dict: Parsed JSON content

    Raises:
        FileNotFoundError: If the file does not exist in S3
        json.JSONDecodeError: If the file content is not valid JSON
        Exception: For other unexpected errors
    """
    s3 = boto3.client('s3')
    logging.info(f"Attempting to read S3 file '{scoringDataFile}' from bucket '{bucket_name}'")

    try:
        # Attempt to get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=scoringDataFile)
        content = response['Body'].read().decode('utf-8')  # Decode bytes to string
        logging.info(f"Successfully retrieved file '{scoringDataFile}' from S3")
        
        # Parse JSON content
        data = json.loads(content)
        return data

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logging.error(f"File '{scoringDataFile}' does not exist in bucket '{bucket_name}'")
            return {
                "predictionScores": "",
                "returnCode": 1,
                "error": f"S3 file '{scoringDataFile}' not found in bucket '{bucket_name}'"
            }
        else:
            logging.exception(f"A client error occurred while accessing S3: {e}")
            return {
                "predictionScores": "",
                "returnCode": 1,
                "error": f"A client error occurred while accessing S3: {e}"
            }
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file '{scoringDataFile}' from bucket '{bucket_name}': {e}")
        return {
                "predictionScores": "",
                "returnCode": 1,
                "error": f"Invalid JSON in file '{scoringDataFile}' from bucket '{bucket_name}': {e}"
            }
    except Exception as e:
        logging.exception(f"Unexpected error reading file '{scoringDataFile}' from S3")
        return {
                "predictionScores": "",
                "returnCode": 1,
                "error": f"Unexpected error reading file '{scoringDataFile}' from S3"
            }
    finally:
        logging.info(f"Finished attempting to read file '{scoringDataFile}' from S3")


def file_exists_in_s3(bucket_name, scoringDataFile):
    """
    Checks if a file exists in an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket
        scoringDataFile (str): Name of the file/key to check

    Returns:
        bool: True if the file exists, False if it does not exist

    Raises:
        Exception: For unexpected errors accessing S3
    """
    s3 = boto3.client('s3')
    logging.info(f"Checking existence of S3 file '{scoringDataFile}' in bucket '{bucket_name}'")

    try:
        # Try to retrieve metadata for the object
        s3.head_object(Bucket=bucket_name, Key=scoringDataFile)
        logging.info(f"File '{scoringDataFile}' exists in bucket '{bucket_name}'")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404' or error_code == 'NoSuchKey':
            logging.warning(f"File '{scoringDataFile}' does not exist in bucket '{bucket_name}'")
            return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"File '{scoringDataFile}' does not exist in bucket '{bucket_name}'"
                }
        else:
            logging.exception(f"ClientError when checking file '{scoringDataFile}' in bucket '{bucket_name}': {e}")
            return {
                "predictionScores": "",
                "returnCode": 1,
                "error": str(e)
            }
    except Exception as e:
        logging.exception(f"Unexpected error when checking file '{scoringDataFile}' in bucket '{bucket_name}': {e}")
        return {
            "predictionScores": "",
            "returnCode": 1,
            "error": str(e)
        }


def main(modelName,scoringDataFile,bucket_name,useS3Bucket,models):
    
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
    
        logging.info("Using cached model '%s' (%s)", modelName,models["model"].__class__.__name__)
        
        logging.info(f'executing main method {startTime}')

        s3 = boto3.client('s3')
                    

        logging.info(f"=" * 70)
        logging.info("CART Scoring Pipeline Verification")
        logging.info(f"=" * 70)   

        base_dir = Path(__file__).parent
        errors = []
        logging.info(f'base_dir: {base_dir}');       
                                                                        
        # 1. Retrieve Cached models
        logging.info(f"\n1. Retrieving models for :{modelName}")
        try:
            if modelName == "feedback":
                fb_model = models.get("model") 
                fb_cal = models.get("calibrator") 
                fb_scaler = models.get("scaler") 
            else :
                io_model = models.get("model") 
                io_cal = models.get("calibrator") 
                io_scaler = models.get("scaler") 
                logging.info(f"OK -  models retrieved for {modelName}")
        except Exception as e:
            logging.exception(f"Error retrieving models: {e}")
            return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"Model retrieval failed: {e}"
                }
        
        # 2. Load Data
        try:
            logging.info(f"The bucket flag '{useS3Bucket}' exists in the bucket '{bucket_name}'.")
            if useS3Bucket == "false":
                logging.info('reading the file from File folder:')
                articles = read_json_file(scoringDataFile)
                df = pd.DataFrame(articles)
            elif useS3Bucket == "true" and file_exists_in_s3(bucket_name, scoringDataFile):
                logging.info(f"The file '{scoringDataFile}' exists in the bucket '{bucket_name}'.")
                # Proceed to read the file from S3
                articles = read_file_from_s3(bucket_name, scoringDataFile)
            else:
                logging.warning(f"Invalid useS3Bucket flag value: {useS3Bucket}")
                return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"Invalid useS3Bucket flag: {useS3Bucket}"
                }

            if not articles:
                logging.warning(f"No data loaded from file '{scoringDataFile}'")
                return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": "No data loaded from scoringDataFile "
                }
			
            df = pd.DataFrame(articles)
        except Exception as e:
            logging.exception(f"Error loading input data: {e}")
            return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"Data loading failed: {e}"
                }
        
            # 3. Preprocessing
        try:
            if modelName == "feedback" :
                logging.info("\n4. Preprocessing for Feedback+Identity model...")
                df_fb = df.copy()
                for feat in FEEDBACK_IDENTITY_BASE_FEATURES:
                    if feat not in df_fb.columns:
                        df_fb[feat] = 0
                    df_fb[feat] = df_fb[feat].fillna(0)
                df_fb = compute_derived_features_feedback_identity(df_fb)
                logging.info(f"   OK - {len(FEEDBACK_IDENTITY_FEATURES)} features prepared")

            elif modelName =="identity" :
                logging.info("\n5. Preprocessing for Identity-Only model...")
                df_io = df.copy()
                for feat in IDENTITY_ONLY_BASE_FEATURES:
                    if feat not in df_io.columns:
                        df_io[feat] = 0
                    df_io[feat] = df_io[feat].fillna(0)
                df_io = compute_derived_features_identity_only(df_io)
                logging.info(f"   OK - {len(IDENTITY_ONLY_FEATURES)} features prepared")
            else:
                logging.warning(f"Invalid modelName: {modelName}")
                return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"Invalid modelName: {modelName}"
                }
        
        except Exception as e:
            logging.exception(f"Error during preprocessing: {e}")
            return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"Preprocessing failed: {e}"
                }
            # 4. Scoring
        try:    
            logging.info("\n6. Scoring articles...")
            if modelName =="feedback" :
                X_fb = fb_scaler.transform(df_fb[FEEDBACK_IDENTITY_FEATURES].values)
                raw_fb = fb_model.predict_proba(X_fb)[:, 1]
                cal_fb = fb_cal.predict(raw_fb.reshape(-1, 1))
                score_fb = cal_fb * 100
                # Prepare the output
                # Make a dictionary for each row
                scoring_output = [
                    {'id': int(article_id), 'scoreTotal': float(score)}
                    for article_id, score in zip(df_fb['articleId'], score_fb)
                ]
                logging.info("scoring_output:\n%s",scoring_output)
            
            elif modelName == "identity":    
                X_io = io_scaler.transform(df_io[IDENTITY_ONLY_FEATURES].values)
                raw_io = io_model.predict_proba(X_io)[:, 1]
                cal_io = io_cal.predict(raw_io)
                score_io = cal_io * 100
            
                # Prepare the output
                # Make a dictionary for each row
                scoring_output = [
                    {'id': int(article_id), 'scoreTotal': float(score)}
                    for article_id, score in zip(df_io['articleId'], score_io)
                ]
                logging.info("scoring_output:\n%s",scoring_output)
                logging.info("   OK - Predictions generated")
            return {
                    "predictionScores": scoring_output,
                    "returnCode": 0,
                    "error": ""
                }

        except Exception as e:
            logging.exception(f"Error during scoring: {e}")
            return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"Scoring failed: {e}"
                }
    
    except Exception as e:
        logging.exception(f"Unexpected error in main(): {e}")
        return {
                    "predictionScores": "",
                    "returnCode": 1,
                    "error": f"Unexpected failure: {e}"
                }

