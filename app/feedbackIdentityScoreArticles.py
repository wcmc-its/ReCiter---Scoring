import json
import sys
import pandas as pd
import numpy as np
import time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO, WARNING, and ERROR messages

import joblib
import logging
import argparse
import boto3
from botocore.exceptions import NoCredentialsError

import warnings
warnings.filterwarnings('ignore')
try:
    from urllib3.exceptions import SNIMissingWarning
except ImportError:
    # Handle the absence or use an alternative
    SNIMissingWarning = None


# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(logging.StreamHandler(sys.stderr))


startTime = time.perf_counter();

# Set up argument parsing
parser = argparse.ArgumentParser(description="Read a JSON file.")
parser.add_argument('file_name', type=str, help='The name of the JSON file to read')
parser.add_argument('bucket_name', type=str, help='The name of the S3 bucket')
parser.add_argument('useS3Bucket', type=str, help='Flag whether to use S3 Bucket or not')

# Parse the arguments
args = parser.parse_args()
s3 = boto3.client('s3')

def read_json_file(file_name):
    try:
        logging.info(f"Filename for input processing. {file_name}")
        file_path = os.path.join("/var/task/data", file_name)
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


def compute_derived_features(df):
    """
    Compute the 4 derived features required by the feedbackIdentity model.
    These features are computed from base features and capture higher-level patterns.
    """
    # identityStrength: 1.0 if any strong identity signal present, else 0.0
    # Strong signals: email match, ORCID match, or institutional affiliation match
    df['identityStrength'] = df.apply(
        lambda row: 1.0 if (
            row.get('emailMatchScore', 0) > 0 or
            row.get('feedbackScoreOrcid', 0) > 0 or
            row.get('targetAuthorInstitutionalAffiliationMatchTypeScore', 0) > 0
        ) else 0.0, axis=1)

    # acceptanceRateLowerBound: Wilson score lower bound on acceptance rate
    # This gives a conservative estimate of the true acceptance rate
    n = df['countAccepted'] + df['countRejected']
    p = df['countAccepted'] / n.replace(0, 1)  # Avoid division by zero
    z = 1.96  # 95% confidence
    denominator = 1 + z*z/n
    numerator = p + z*z/(2*n) - z*np.sqrt((p*(1-p) + z*z/(4*n))/n)
    df['acceptanceRateLowerBound'] = (numerator / denominator).fillna(0.5)

    # feedbackConfidence: log(1 + total feedback count)
    # Higher values indicate more historical data to learn from
    df['feedbackConfidence'] = np.log1p(df['countAccepted'] + df['countRejected'])

    # uncertainRejectionRisk: risk score for uncertain cases with high rejection
    # High when identity is weak AND acceptance rate is low
    df['uncertainRejectionRisk'] = (1 - df['identityStrength']) * (1 - df['acceptanceRateLowerBound'])

    return df


logging.info(f"The bucket flag '{args.useS3Bucket}' exists in the bucket '{args.bucket_name}'.")
if args.useS3Bucket == "false":
    logging.info('reading the file from File folder:')
    data = read_json_file(args.file_name)
    if data is not None:
        logging.info(data)
elif args.useS3Bucket == "true" and file_exists_in_s3(args.bucket_name, args.file_name):
    logging.info(f"The file '{args.file_name}' exists in the bucket '{args.bucket_name}'.")
    # Proceed to read the file from S3
    data = read_file_from_s3(args.bucket_name, args.file_name)
    if data is not None:
        logging.info(data)
else:
    logging.info(f"Error: The file '{args.file_name}' does not exist locally and the file '{args.file_name}' does not exist in the bucket '{args.bucket_name}' and useS3Bucket is '{args.useS3Bucket}'.")


model = joblib.load('feedbackIdentityModel.joblib')
calibrator = joblib.load('feedbackIdentityCalibrator.joblib')
scaler = joblib.load('feedbackIdentityScaler.joblib')

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)

# Fill missing base features with 0
base_feature_columns = [
    'feedbackScoreCites', 'feedbackScoreCoAuthorName', 'feedbackScoreEmail',
    'feedbackScoreInstitution', 'feedbackScoreJournal', 'feedbackScoreJournalSubField',
    'feedbackScoreKeyword', 'feedbackScoreOrcid', 'feedbackScoreOrcidCoAuthor',
    'feedbackScoreOrganization', 'feedbackScoreTargetAuthorName', 'feedbackScoreYear',
    'articleCountScore', 'authorCountScore', 'discrepancyDegreeYearScore', 'emailMatchScore',
    'genderScoreIdentityArticleDiscrepancy', 'grantMatchScore', 'journalSubfieldScore',
    'nameMatchFirstScore', 'nameMatchLastScore', 'nameMatchMiddleScore',
    'nameMatchModifierScore', 'organizationalUnitMatchingScore',
    'scopusNonTargetAuthorInstitutionalAffiliationScore',
    'targetAuthorInstitutionalAffiliationMatchTypeScore',
    'pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore',
    'relationshipPositiveMatchScore', 'relationshipNegativeMatchScore',
    'relationshipIdentityCount', 'countAccepted', 'countRejected'
]

for col in base_feature_columns:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0)

# Compute derived features (required by the model)
df = compute_derived_features(df)

# Full feature list in the exact order expected by the model
feature_columns = [
    'feedbackScoreCites', 'feedbackScoreCoAuthorName', 'feedbackScoreEmail',
    'feedbackScoreInstitution', 'feedbackScoreJournal', 'feedbackScoreJournalSubField',
    'feedbackScoreKeyword', 'feedbackScoreOrcid', 'feedbackScoreOrcidCoAuthor',
    'feedbackScoreOrganization', 'feedbackScoreTargetAuthorName', 'feedbackScoreYear',
    'articleCountScore', 'authorCountScore', 'discrepancyDegreeYearScore', 'emailMatchScore',
    'genderScoreIdentityArticleDiscrepancy', 'grantMatchScore', 'journalSubfieldScore',
    'nameMatchFirstScore', 'nameMatchLastScore', 'nameMatchMiddleScore',
    'nameMatchModifierScore', 'organizationalUnitMatchingScore',
    'scopusNonTargetAuthorInstitutionalAffiliationScore',
    'targetAuthorInstitutionalAffiliationMatchTypeScore',
    'pubmedTargetAuthorInstitutionalAffiliationMatchTypeScore',
    'relationshipPositiveMatchScore', 'relationshipNegativeMatchScore',
    'relationshipIdentityCount', 'countAccepted', 'countRejected',
    'identityStrength', 'acceptanceRateLowerBound', 'feedbackConfidence', 'uncertainRejectionRisk'
]

X = df[feature_columns]

# Scale the features
features_scaled = scaler.transform(X)

# Run the predictions
raw_probs = model.predict_proba(features_scaled)[:, 1]
calibrated_probs = calibrator.predict(raw_probs)

# Prepare the output
scoring_output = []
for idx, row in df.iterrows():
    id_value = row['articleId']
    inverted_score = float(calibrated_probs[idx]) * 100
    scoring_output.append({'id': id_value, 'scoreTotal': inverted_score})

# Print the scoring output as JSON to return it to the Java process
logging.info(f"script execution completed successfully: {scoring_output}")
endTime = time.perf_counter()
logging.info(f"Elapsed time: {endTime - startTime:.3f} seconds")
print(json.dumps(scoring_output))
