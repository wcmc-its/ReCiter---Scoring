import subprocess
import logging
import sys


# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(logging.StreamHandler(sys.stderr))

def lambda_handler(event, context):
    
    try:
        scriptFile = event.get('scriptFile','')
        logging.info(f"scriptFile. {scriptFile}")
        fileName = event.get('inputDataFile', '')
        logging.info(f"fileName. {fileName}")
        useS3Bucket = event.get('useS3Bucket', 'false')
        logging.info(f"useS3Bucket. {useS3Bucket}")
        bucket_name = event.get('bucket_name','')
        logging.info(f"bucket_name. {bucket_name}")
        pythonCommandName = event.get('pythonCommandName','python')
        logging.info(f"pythonCommandName. {pythonCommandName}")
    except Exception as e:
        logger.error(f"Exception occurred: {e}")

    if scriptFile.strip():
        logging.info('coming into feedbackIdentity')
        result = subprocess.run([pythonCommandName, scriptFile,fileName,bucket_name,useS3Bucket],stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(f"result {result}") 
        return {
            "authorshiplikelihoodScores": result.stdout.strip(),
            "returncode": result.returncode 
        }
