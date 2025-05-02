import pandas as pd
import joblib
from keras.models import load_model
import json
import sys
import logging
import os
import keras
# Set up logging configuration
logging.basicConfig(filename='script.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Current Working Directory: %s", os.getcwd())

def read_json_file(file_path):
    try:
        logging.info("Script is starting.")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data  # Return the loaded JSON data
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")
        sys.exit(1)  # Exit if there's an error in loading the data

if __name__ == "__main__":
    # Load JSON data from the specified file
    json_data = read_json_file('output.json')  # Ensure this path is correct when called from Java
    print("All data fetched successfully from JSON!")
    logging.info(f"Data read from JSON file: {json_data}")
    print(keras.__version__)
    # Load JSON data into a DataFrame
    try:
        df_all = pd.DataFrame(json_data)  # Convert JSON data to DataFrame
        logging.info(f"Data read from JSON file: {df_all}")
    except Exception as e:
        print(f"Error converting JSON to DataFrame: {e}")
        sys.exit(1)  # Exit if there's an error in loading the data

    # Preprocessing all data
    df_all = df_all.dropna()
    logging.info(f"Data read from JSON file1: {df_all}")
    # Load the scaler and model (ensure the paths are correct)
    scaler = joblib.load('StandardScaler.save')  # Load your scaler
    logging.info(f"Mean: {scaler.mean_}")
    logging.info(f"Variance: {scaler.var_}")
    logging.info(f"Scale: {scaler.scale_}")
    
    model = load_model('feedbackScoringModel 1.keras')  # Load your Keras model
    model.summary()

    # Get the model configuration
    config = model.get_config()
    print(config)  # This will print the configuration as a dictionary

    

    for layer in model.layers:
        print(layer.name, layer.output_shape)

    weights = model.layers[0].get_weights()  # Replace 0 with the index of the layer you want to inspect
    logging.info(weights)


    # Transform the features for prediction
    X_all = scaler.transform(df_all.drop(['articleId'], axis=1))
    logging.info(f"Data read from JSON file2: {X_all}")
    # Generating scores using the model
    scores = model.predict(X_all).flatten()
    logging.info(f"scores: {scores}")
    print("Scores:", scores)

    # Invert the scores
    inverted_scores = 1 - scores

    # Proceed to save the inverted scores to CSV
    score_data = pd.DataFrame({'articleId': df_all['articleId'], 'scoreModificationProposed': inverted_scores})
    score_data.to_csv('output.csv', index=False)
    print("Inverted scores saved to 'output.csv'.")
    logging.info("Inverted scores saved to 'output.csv'.")
