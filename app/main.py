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
        modelName = event.get('modelName','')
        logging.info(f"modelName. {modelName}")
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
    

        if scriptFile.strip():
            result = subprocess.run([pythonCommandName, scriptFile,modelName,fileName,bucket_name,useS3Bucket],  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            # Optional: Log stderr for debugging
            if stderr:
                logging.warning(f"logging information from the {scriptFile} : {stderr}")
            return {
                "authorshiplikelihoodScores": stdout,
                "returncode": result.returncode 
            }
        
    except Exception as e:
        logging.error(f"Failed to run subprocess: {e}")
        return {
            "error": str(e)
        }