import subprocess
import logging
import os
# Set up logging configuration
logging.basicConfig(filename='ReCiterScoring.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def lambda_handler(event, context):
   
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
        logging.info('coming into feedbackIdentity')
        result = subprocess.run([pythonCommandName, scriptFile,fileName,bucket_name,useS3Bucket],stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(f"result {result}") 
        return {
            "authorshiplikelihoodScores": result.stdout.strip(),
            "returncode": result.returncode 
        }
