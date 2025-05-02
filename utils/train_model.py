import json
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


# Load data from JSON file
with open('output.json') as f:
    data = json.load(f)

# Convert JSON data to a DataFrame
df = pd.json_normalize(data)

# Separate features and labels
X = df[['feedbackScoreCites', 'feedbackScoreCoAuthorName', 'feedbackScoreEmail','feedbackScoreInstitution','feedbackScoreJournal','feedbackScoreJournalSubField','feedbackScoreKeyword','feedbackScoreOrcid','feedbackScoreOrcidCoAuthor','feedbackScoreOrganization','feedbackScoreTargetAuthorName','feedbackScoreYear']].values  # Features
y = df['articleId'].values                    # Labels

# Create a simple model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),  # Input layer
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create lists to save scalar values
loss_values = []
accuracy_values = []

# Train the model and save scalar values
epochs = 10
    history = model.fit(X, y, epochs=1, batch_size=2, verbose=1)
for epoch in range(epochs):
    
    # Save loss and accuracy
    loss_values.append(history.history['loss'][0])
    accuracy_values.append(history.history['accuracy'][0])

# Save the model
model.save('my_model.keras')
print("Model saved as 'my_model.keras'")

# Save scalar values to a .npy file
np.save('loss_values.save', loss_values)
np.save('accuracy_values.save', accuracy_values)
print("Scalar values saved as 'loss_values.save' and 'accuracy_values.save'")
