import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
import joblib
import logging
import sys
from verify_setup import main as run_main
from pathlib import Path



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True
)

# Directory containing your models
base_dir = Path(__file__).parent / "models"


# Module-level cache
cached_models = {}

try:
    cached_models["feedback"] = {
        "model": joblib.load(base_dir / "feedbackIdentityModel.joblib"),
        "calibrator": joblib.load(base_dir / "feedbackIdentityCalibrator.joblib"),
        "scaler": joblib.load(base_dir / "feedbackIdentityScaler.joblib")
    }

    # Disable multiprocessing if supported
    if hasattr(cached_models["feedback"]["model"], "n_jobs"):
        cached_models["feedback"]["model"].set_params(n_jobs=1)


    cached_models["identity"] = {
        "model": joblib.load(base_dir / "identityOnlyModel.joblib"),
        "calibrator": joblib.load(base_dir / "identityOnlyCalibrator.joblib"),
        "scaler": joblib.load(base_dir / "identityOnlyScaler.joblib")
    }
    # Disable multiprocessing if supported
    if hasattr(cached_models["identity"]["model"], "n_jobs"):
        cached_models["identity"]["model"].set_params(n_jobs=1)

    logging.info("All models loaded successfully at cold start")
except Exception as e:
    logging.exception(f"Error loading models: {e}")
    cached_models = None

def lambda_handler(event, context):
    
    if cached_models is None:
        return {
            "predictionScores": "",
            "returnCode": 1,
            "error": "Models not loaded"
        }

    try:
        modelName = event.get('modelName','')
        scoringDataFile = event.get('scoringDataFile', '')
        useS3Bucket = event.get('useS3Bucket', 'false')
        bucket_name = event.get('bucket_name','')
        logging.info(
                        f"modelName: {modelName}, inputDateFileName: {scoringDataFile}\n"
                        f"useS3Bucket: {useS3Bucket}, bucketName: {bucket_name} "
                        
                    )

        models = cached_models.get(modelName)
        logging.info("Using cached model '%s' (%s)", modelName,models["model"].__class__.__name__)

        if not models:
            return {
                "predictionScores": "",
                "returnCode": 1,
                "error": f"Unknown model type: {modelName}"
            }


        # Return error if inputData is not provided
        if not scoringDataFile or not scoringDataFile.strip():
            logging.warning("InputDateFileName is empty or blank")
            return {
                "predictionScores": "",
                "returnCode": 1,
                "error": "Invalid inputDateFileName"
            }

        # Proceed after successful validation    
        result = run_main(modelName,scoringDataFile,bucket_name,useS3Bucket,models)
        logging.info(f"result************:{result}")
        returnCode = result.get("returnCode")
        predictionScores = result.get("predictionScores")
        error = result.get("error")
        logging.info(f"returnCode************:{returnCode}")
        logging.info(f"predictionScores************:{predictionScores}")
        logging.info(f"error***********:{error}")
        return {
            "predictionScores": predictionScores,
            "returnCode": returnCode,
            "error": error, 
        }
        
    except Exception as e:
        logging.error(f"Failed to run verify_setup script: {e}")
        return {
            "predictionScores": "",
            "returnCode": 1,
            "error": str(e)
        }
