import subprocess
import logging
import sys


import logging

#logging.basicConfig(
#    level=logging.INFO,  # INFO and higher (WARNING, ERROR, CRITICAL) will show
#    format='%(asctime)s - %(levelname)s - %(message)s'
#)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True
)


logging.info("This is info")
logging.warning("This is a warning")
logging.error("This is an error")


# Set up logging configuration
#logger = logging.getLogger()
#logger.setLevel(logging.INFO)
#logger.handlers.clear()
#logger.addHandler(logging.StreamHandler(sys.stderr))
#logger.addHandler(logging.StreamHandler(sys.stdout))


def lambda_handler(event, context):
    
    try:
        modelName = event.get('modelName','')
     #   logging.info(f"modelName. {modelName}")
        scriptFile = event.get('scriptFile','')
     #   logging.info(f"scriptFile. {scriptFile}")
        fileName = event.get('inputDataFile', '')
     #   logging.info(f"fileName. {fileName}")
        useS3Bucket = event.get('useS3Bucket', 'false')
     #   logging.info(f"useS3Bucket. {useS3Bucket}")
        bucket_name = event.get('bucket_name','')
     #   logging.info(f"bucket_name. {bucket_name}")
        pythonCommandName = event.get('pythonCommandName','python')
     #   logging.info(f"pythonCommandName. {pythonCommandName}")
        logging.info(
                        f"modelName: {modelName}, scriptFile: {scriptFile}, fileName: {fileName}\n"
                        f"useS3Bucket: {useS3Bucket}, bucketName: {bucket_name}, "
                        f"pythonCommandName: {pythonCommandName}"
                    )
    

        if scriptFile.strip():
            logging.info("About to run subprocess");
            result = subprocess.run([pythonCommandName, scriptFile,modelName,fileName,bucket_name,useS3Bucket],  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logging.info("subprocess completed run");
            logging.info(f"result************:{result}")
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            logging.info(f"result************:{result}")
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
