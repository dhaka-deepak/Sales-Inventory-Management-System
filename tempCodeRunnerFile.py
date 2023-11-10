from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('trained_model.h5')  # Replace with the actual path to your trained model file
scaler = StandardScaler()  # Assuming you saved the scaler during training, adjust if needed
scaler.fit(np.random.rand(100, 1))  # Fitting a random data to avoid warnings, replace with actual data

# Load the sales data
df = pd.read_excel('sales_data.xlsx')  # Replace with the actual path to your sales data file

# Remove trailing spaces from column names
df.columns = df.columns.str.strip()

# Extract relevant columns
data = df[['ProductID', 'Quantity', 'Date and Time']]

# Convert 'Date and Time' to datetime and set it as index
data['Date and Time'] = pd.to_datetime(data['Date and Time'])
data.set_index('Date and Time', inplace=True)

# Resample the data to get total sales per day
sales_per_day = data.groupby(['ProductID']).resample('D').sum()['Quantity']

# Function to make predictions for a specific date and product
def predict_quantity(product_id, target_date):
    # Extract sales data for the specified product
    product_sales = sales_per_day.loc[product_id]

    # Extract the sequence leading up to the target date
    target_sequence = product_sales.loc[:target_date].tail(sequence_length)

    # If there are not enough historical data, return an error message
    if len(target_sequence) < sequence_length:
        return "Insufficient historical data for prediction"

    # Normalize the data
    scaled_data = scaler.transform(np.array(target_sequence).reshape(1, -1))

    # Reshape the data for CNN
    input_data = scaled_data.reshape((1, sequence_length, 1))

    # Make the prediction
    predicted_quantity = model.predict(input_data)

    # Inverse transform to get the actual predicted quantity
    predicted_quantity = scaler.inverse_transform(predicted_quantity.reshape(-1, 1))

    # Return the result
    return predicted_quantity[0, 0]

# API route for predictions
@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()

    product_id = data.get('product_id')
    target_time = pd.Timestamp(data.get('target_time'))

    try:
        predicted_quantity = predict_quantity(product_id, target_time)
        return jsonify({'product_id': product_id, 'target_time': str(target_time), 'predicted_quantity': predicted_quantity})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
