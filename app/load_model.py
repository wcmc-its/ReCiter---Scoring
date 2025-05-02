from tensorflow import keras
import numpy as np

# Load the model
loaded_model = keras.models.load_model('my_model.keras')

# Example new data for predictions
new_data = np.array([[0.6, 0.8], [0.1, 0.4]])
predictions = loaded_model.predict()

print("Predictions:")
print(predictions)
