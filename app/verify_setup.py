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

logging.basicConfig(filename='script.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')    
logging.info("Current Working Directory: %s", os.getcwd())

import warnings
warnings.filterwarnings('ignore')

TOLERANCE = 0.1  # Acceptable difference for authorshipLikelihoodScore (0-100 scale)


def read_json_file(file_name):
    try:
        logging.info(f"Filename for input processing. {file_name}")
        file_path = os.path.join("/var/task/data", file_name)
        #file_path = os.path.join("app", file_name)
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
# logging.info()
# logging.info("CART Scoring Pipeline Verification")
# logging.info("=" * 70)

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
#if args.file_name == "null" or args.file_name == "" :
#    args.file_name = args.modelName + "_input_data_" + args.cwid +".json"; 
#    logging.info(f'file_name*********: {args.file_name}')
if args.useS3Bucket == "false":
    logging.info('reading the file from File folder:')
    articles = read_json_file(args.file_name)
   # logging.info(f"articles******** {articles}")
    df = pd.DataFrame(articles)
#if articles is not None:
#        logging.info(articles)
#el
if args.useS3Bucket == "true" and file_exists_in_s3(args.bucket_name, args.file_name):
    logging.info(f"The file '{args.file_name}' exists in the bucket '{args.bucket_name}'.")
    # Proceed to read the file from S3
    articles = read_file_from_s3(args.bucket_name, args.file_name)
    df = pd.DataFrame(articles)
    if articles is not None:
        logging.info(articles)
    else:
        logging.info(f"Error: The file '{args.file_name}' does not exist locally and the file '{args.file_name}' does not exist in the bucket '{args.bucket_name}' and useS3Bucket is '{args.useS3Bucket}'.")
	
    # 2. Load sample input
    #logging.info("\n2. Loading sample input...")
    #try:
    #    with open(base_dir / "sample_input_ajg9004.json") as f:
    #        articles = json.load(f)
    #    df = pd.DataFrame(articles)
    #    logging.info(f"   OK - Loaded {len(df)} articles")
    #except Exception as e:
    #    logging.info(f"   FAIL - {e}")
    #    sys.exit(1)

    # 3. Load expected outputs
    #logging.info("\n3. Loading expected outputs...")
    #try:
    #    with open(base_dir / "expected_outputs_ajg9004.json") as f:
    #        expected = json.load(f)
			  
    #    logging.info(f"   OK - Loaded expected outputs for {len(expected['articles'])} articles")
		 
    #except Exception as e:
    #    logging.info(f"   FAIL - {e}")
    #    sys.exit(1)
#expected_output_file_name = args.modelName + "_expected_output_" + args.cwid +".json"; 
#logging.info(f'expected_output_file_name {expected_output_file_name}')
#if args.useS3Bucket == "false":
#    logging.info('reading the file from File folder:')
#    expected = read_json_file(expected_output_file_name)
#    logging.info(f"expected data {expected}")

#if expected is not None:
#    logging.info(expected)
#elif args.useS3Bucket == "true" and file_exists_in_s3(args.bucket_name, expected_output_file_name):
#    logging.info(f"The file '{expected_output_file_name}' exists in the bucket '{args.bucket_name}'.")
# Proceed to read the file from S3
#    expected = read_file_from_s3(args.bucket_name, expected_output_file_name)
#    if expected is not None:
#        logging.info(expected)
#    else:
#        logging.info(f"Error: The file '{expected_output_file_name}' does not exist locally and the file '{expected_output_file_name}' does not exist in the bucket '{args.bucket_name}' and useS3Bucket is '{args.useS3Bucket}'.")

    # 4. Preprocess for Feedback+Identity
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

# 7. Verify against expected
#logging.info("\n7. Verifying against expected outputs...")
#n_checked = 0
#n_passed = 0

#logging.info(f'expected["articles"]: {expected["articles"]}')
#for i, exp_art in enumerate(expected["articles"]):
 #   n_checked += 1

 #   if args.modelName == "feedback" :
    #    logging.info('inside feedback model')
  #      exp_fb = exp_art["scores"]["feedback_identity"]["authorshipLikelihoodScore"]
  #      diff_fb = abs(score_fb[i] - exp_fb)

  #  elif args.modelName == "identity" :
  #      logging.info('inside Identity model')
  #      exp_io = exp_art["scores"]["identity_only"]["authorshipLikelihoodScore"]
  #      diff_io = abs(score_io[i] - exp_io)

  #  if args.modelName =="feedback" and diff_fb > TOLERANCE:
      #  logging.info('inside feedback model1')
   #     errors.append(
   #         f"Article {i} (id={exp_art['articleId']}): "
   #         f"FB expected={exp_fb:.2f}, got={score_fb[i]:.2f}, diff={diff_fb:.2f}"
   #     )
   # elif args.modelName =="identity" and diff_io > TOLERANCE:
   #     logging.info('inside identity model1')
   #     errors.append(
   #         f"Article {i} (id={exp_art['articleId']}): "
   #         f"IO expected={exp_io:.2f}, got={score_io[i]:.2f}, diff={diff_io:.2f}"
   #     )
    #else:
    #    n_passed += 1

#logging.info(f"   Checked: {n_checked} articles")
#logging.info(f"   Passed:  {n_passed} articles")
#logging.info(f"   Failed:  {len(errors)} articles")

# 8. Summary
#logging.info("\n" + "=" * 70)
#if errors:
#    logging.info("VERIFICATION FAILED")
#    logging.info("=" * 70)
#    logging.info("\nFirst 10 errors:")
#    for err in errors[:10]:
#        logging.info(f"  - {err}")
#    if len(errors) > 10:
#        logging.info(f"  ... and {len(errors) - 10} more errors")
#    sys.exit(1)
#else:
#    logging.info("VERIFICATION PASSED")
#    logging.info("=" * 70)
#    logging.info("\nAll authorshipLikelihoodScore values match within tolerance.")
#    logging.info(f"  Tolerance: ±{TOLERANCE} (0-100 scale)")
#    logging.info(f"  Articles verified: {n_checked}")

    # Show sample outputs
#    logging.info("\nSample outputs (first 5 articles):")
#    if args.debug:
        # Debug mode: show all intermediate values
                
#        logging.info(f"{'Idx':>4} {'ArticleID':>12} {'IO_score':>9} {'IO_raw':>8} {'IO_cal':>8} "
#                f"{'FB_score':>9} {'FB_raw':>8} {'FB_cal':>8} {'Label':<10}")
            
#        logging.info("-" * 95)
#        for i in range(5):
#            art = expected["articles"][i]
#            logging.info(
#                f"{i:>4} {art['articleId']:>12} "
#                f"{score_io[i]:>9.2f} {raw_io[i]:>8.4f} {cal_io[i]:>8.4f} "
#                f"{score_fb[i]:>9.2f} {raw_fb[i]:>8.4f} {cal_fb[i]:>8.4f} "
#                f"{art['userAssertion']:<10}"
#            )
#        logging.info("\n(Use --debug to see intermediate raw/calibrated probabilities)")
#    else:
        # Standard mode: just show final scores
                
#        logging.info(f"{'Idx':>4} {'ArticleID':>12} {'IO_score':>12} {'FB_score':>12} {'Label':<10}")
            
#        logging.info("-" * 58)
#        for i in range(5):
#            art = expected["articles"][i]
#            logging.info(
#                f"{i:>4} {art['articleId']:>12} "
#                f"{score_io[i]:>12.2f} {score_fb[i]:>12.2f} "
#                f"{art['userAssertion']:<10}"
#            )
#        logging.info("\n(Use --debug to see intermediate raw/calibrated probabilities)")

#    sys.exit(0)


if __name__ == "__main__":
    main()
