import subprocess
import logging
import os,sys
# Set up logging configuration
logging.basicConfig(filename='ReCiterScoring.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)])

logger = logging.getLogger(__name__)

def lambda_handler(event, context):
   
	
    scriptFile = event.get('scriptFile','')
    print("stdout: test print")
    print("stderr: test print", file=sys.stderr)
    logging.info("Logging INFO test")
    logging.error("Logging ERROR test")
    os.system("echo SYSTEM call executed")
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
