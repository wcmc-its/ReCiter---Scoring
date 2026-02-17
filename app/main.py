import subprocess
import logging
import sys


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True
)

def lambda_handler(event, context):
    
    try:
        modelName = event.get('modelName','')
        scriptFile = event.get('scriptFile','')
        fileName = event.get('inputDataFile', '')
        useS3Bucket = event.get('useS3Bucket', 'false')
        bucket_name = event.get('bucket_name','')
        pythonCommandName = event.get('pythonCommandName','python')
        logging.info(
                        f"modelName: {modelName}, scriptFile: {scriptFile}, fileName: {fileName}\n"
                        f"useS3Bucket: {useS3Bucket}, bucketName: {bucket_name}, "
                        f"pythonCommandName: {pythonCommandName}"
                    )
    

        if scriptFile.strip():
            result = subprocess.run([pythonCommandName, scriptFile,modelName,fileName,bucket_name,useS3Bucket],  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logging.info(f"result************:{result}")
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            logging.info(f"stdout************:{stdout}")
            logging.info(f"stderr************:{stderr}")
            # Optional: Log stderr for debugging
            if stderr:
                logging.warning(f"logging information from the {scriptFile} : {stderr} : ")
            return {
                "authorshiplikelihoodScores": stdout,
                "returncode": result.returncode,
                "stderr": stderr, 
            }
        
    except Exception as e:
        logging.error(f"Failed to run subprocess: {e}")
        return {
            "error": str(e)
        }
