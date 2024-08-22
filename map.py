# import googlemaps
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
# from flask import Flask, request, jsonify

# # Initialize the Google Maps client with your API key
# gmaps = googlemaps.Client(key='AIzaSyCGbYvx87ttSAJV6nGDh_hPfu0ljDOSjfA')

# # Step 1: Fetch Real-Time Traffic Data
# def get_traffic_data(origin, destination):
#     """
#     Fetch real-time traffic data from Google Maps API.

#     Parameters:
#     origin (str): Starting point of the journey.
#     destination (str): End point of the journey.

#     Returns:
#     tuple: travel_time (int), distance (int) or (None, None) if an error occurs
#     """
#     try:
#         now = datetime.now()
#         directions_result = gmaps.directions(origin, destination,
#                                              mode="driving",
#                                              departure_time=now)
        
#         if not directions_result or 'legs' not in directions_result[0]:
#             raise ValueError("Invalid response from API")
        
#         # Extract relevant information
#         travel_time = directions_result[0]['legs'][0]['duration_in_traffic']['value']  # in seconds
#         distance = directions_result[0]['legs'][0]['distance']['value']  # in meters
        
#         return travel_time, distance
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None, None

# # Step 2: Congestion Detection
# def detect_congestion(travel_time, average_travel_time, threshold=1.5):
#     """
#     Detect if there is traffic congestion based on the travel time.

#     Parameters:
#     travel_time (int): Real-time travel time in seconds.
#     average_travel_time (int): Historical average travel time in seconds.
#     threshold (float): Multiplier threshold to detect congestion.

#     Returns:
#     bool: True if congestion is detected, False otherwise.
#     """
#     return travel_time > threshold * average_travel_time

# # Step 3: Train a Prediction Model
# def train_model(data):
#     """
#     Train a linear regression model to predict travel time based on historical data.

#     Parameters:
#     data (pd.DataFrame): Historical traffic data.

#     Returns:
#     model: Trained linear regression model.
#     """
#     X = data[['distance', 'hour']]
#     y = data['travel_time']

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Initialize and train the model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)} seconds")
    
#     return model

# # Step 4: Predict Travel Time
# def predict_travel_time(model, distance, hour):
#     """
#     Predict the travel time using the trained model.

#     Parameters:
#     model: Trained linear regression model.
#     distance (int): Distance of the journey in meters.
#     hour (int): Hour of the day.

#     Returns:
#     float: Predicted travel time in seconds.
#     """
#     return model.predict(np.array([[distance, hour]]))[0]

# # Example historical data for model training
# data = {
#     'travel_time': [600, 800, 1200, 900, 950, 1300, 700],
#     'distance': [5000, 5000, 5000, 5000, 5000, 5000, 5000],
#     'hour': [8, 9, 10, 11, 12, 13, 14]
# }
# df = pd.DataFrame(data)

# # Train the model with historical data
# model = train_model(df)

# # Real-time traffic prediction example
# origin = "mullana"
# destination = "ambala station"
# travel_time, distance = get_traffic_data(origin, destination)
# current_hour = datetime.now().hour
# predicted_travel_time = predict_travel_time(model, distance, current_hour)

# # Calculate average travel time from historical data
# average_travel_time = df['travel_time'].mean()

# # Detect congestion based on predicted travel time
# congestion_detected = detect_congestion(predicted_travel_time, average_travel_time)

# # Output the results
# if congestion_detected:
#     print("Congestion detected!")
# else:
#     print("Traffic is normal.")

# print(f"Real-time travel time: {travel_time / 60:.2f} minutes")
# print(f"Predicted travel time: {predicted_travel_time / 60:.2f} minutes")

# # Step 5: Flask API for Real-Time Traffic Prediction
# app = Flask(__name__)

# @app.route('/predict_traffic', methods=['GET'])
# def predict_traffic():
#     """
#     Flask API endpoint to predict traffic congestion and travel time.

#     Returns:
#     JSON: Contains real-time travel time, predicted travel time, and congestion status.
#     """
#     origin = "mullana ambala"  # Predefined origin
#     destination = "kolkata howrah"  # Predefined destination
    
#     travel_time, distance = get_traffic_data(origin, destination)
    
#     if travel_time is None or distance is None:
#         return jsonify({"error": "Error fetching traffic data"}), 500
    
#     current_hour = datetime.now().hour
#     predicted_travel_time = predict_travel_time(model, distance, current_hour)
#     average_travel_time = df['travel_time'].mean()
#     congestion_detected = detect_congestion(predicted_travel_time, average_travel_time)
    
#     return jsonify({
#         'congestion_detected': str(congestion_detected),  # Convert boolean to string
#         'real_time_travel_time_minutes': travel_time / 60 if travel_time is not None else "N/A",
#         'predicted_travel_time_minutes': predicted_travel_time / 60 if predicted_travel_time is not None else "N/A"
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
import googlemaps
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify

# Initialize the Google Maps client with your API key
gmaps = googlemaps.Client(key='AIzaSyCGbYvx87ttSAJV6nGDh_hPfu0ljDOSjfA')

# Step 1: Fetch Real-Time Traffic Data
def get_traffic_data(origin, destination):
    """
    Fetch real-time traffic data from Google Maps API.

    Parameters:
    origin (str): Starting point of the journey.
    destination (str): End point of the journey.

    Returns:
    tuple: travel_time (int), distance (int) or (None, None) if an error occurs
    """
    try:
        now = datetime.now()
        directions_result = gmaps.directions(origin, destination,
                                             mode="driving",
                                             departure_time=now)
        
        if not directions_result or 'legs' not in directions_result[0]:
            raise ValueError("Invalid response from API")
        
        # Extract relevant information
        travel_time = directions_result[0]['legs'][0]['duration_in_traffic']['value']  # in seconds
        distance = directions_result[0]['legs'][0]['distance']['value']  # in meters
        
        return travel_time, distance
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Step 2: Congestion Detection
def detect_congestion(travel_time, average_travel_time, threshold=1.5):
    """
    Detect if there is traffic congestion based on the travel time.

    Parameters:
    travel_time (int): Real-time travel time in seconds.
    average_travel_time (int): Historical average travel time in seconds.
    threshold (float): Multiplier threshold to detect congestion.

    Returns:
    bool: True if congestion is detected, False otherwise.
    """
    return travel_time > threshold * average_travel_time

# Step 3: Train a Prediction Model
def train_model(data):
    """
    Train a linear regression model to predict travel time based on historical data.

    Parameters:
    data (pd.DataFrame): Historical traffic data.

    Returns:
    model: Trained linear regression model.
    """
    X = data[['distance', 'hour']]
    y = data['travel_time']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)} seconds")
    
    return model

# Step 4: Predict Travel Time
def predict_travel_time(model, distance, hour):
    """
    Predict the travel time using the trained model.

    Parameters:
    model: Trained linear regression model.
    distance (int): Distance of the journey in meters.
    hour (int): Hour of the day.

    Returns:
    float: Predicted travel time in seconds.
    """
    return model.predict(np.array([[distance, hour]]))[0]

# Example historical data for model training
# data = {
#     'travel_time': [600, 800, 1200, 900, 950, 1300, 700, 1000, 850, 1100],
#     'distance': [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#     'hour': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# }
# df = pd.DataFrame(data)
# Example historical data for model training
data = {
    'travel_time': [
        600, 800, 1200, 900, 950, 1300, 700, 1000, 850, 1100,  # 8 AM to 5 PM
        950, 1100, 1450, 850, 1050, 1250, 950, 1150, 1350, 1200,  # Evening
        1000, 1300, 1550, 900, 1150, 1400, 1000, 1250, 1350, 1100,  # Night
        1100, 950, 1200, 950, 1050, 1150, 850, 1000, 1300, 950,  # Weekends
        850, 1000, 1150, 900, 1050, 1200, 1000, 1300, 1450, 850,  # Holidays
        1200, 1000, 1050, 1150, 1400, 950, 1100, 1300, 1200, 1400   # Rush Hours
    ],
    'distance': [
        5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000,
        5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000,
        5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000,
        5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000,
        5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000,
        5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000
    ],
    'hour': [
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 0, 1, 2, 3,
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ]
}
df = pd.DataFrame(data)

# Train the model with historical data
model = train_model(df)

# Real-time traffic prediction example
origin = "noida"
destination = "greater noida"
travel_time, distance = get_traffic_data(origin, destination)
current_hour = datetime.now().hour

# Calculate average travel time from historical data
average_travel_time = df['travel_time'].mean()

if travel_time is not None and distance is not None:
    predicted_travel_time = predict_travel_time(model, distance, current_hour)
    
    # Detect congestion based on predicted travel time
    congestion_detected = detect_congestion(predicted_travel_time, average_travel_time)
    
    # Output the results
    if congestion_detected:
        print("Congestion detected!")
    else:
        print("Traffic is normal.")
    
    print(f"Real-time travel time: {travel_time / 60:.2f} minutes")
    print(f"Predicted travel time: {predicted_travel_time / 60:.2f} minutes")
else:
    print("Error fetching traffic data.")

# Step 5: Flask API for Real-Time Traffic Prediction
app = Flask(__name__)

@app.route('/predict_traffic', methods=['GET'])
def predict_traffic():
    """
    Flask API endpoint to predict traffic congestion and travel time.

    Returns:
    JSON: Contains real-time travel time, predicted travel time, and congestion status.
    """
    origin = "mullana ambala"  # Predefined origin
    destination = "ambala station"  # Predefined destination
    
    travel_time, distance = get_traffic_data(origin, destination)
    
    if travel_time is None or distance is None:
        return jsonify({"error": "Error fetching traffic data"}), 500
    
    current_hour = datetime.now().hour
    predicted_travel_time = predict_travel_time(model, distance, current_hour)
    average_travel_time = df['travel_time'].mean()
    congestion_detected = detect_congestion(predicted_travel_time, average_travel_time)
    
    return jsonify({
        'congestion_detected': str(congestion_detected),  # Convert boolean to string
        'real_time_travel_time_minutes': round(travel_time / 60, 2) if travel_time is not None else "N/A",
        'predicted_travel_time_minutes': round(predicted_travel_time / 60, 2) if predicted_travel_time is not None else "N/A"
    })

if __name__ == '__main__':
    app.run(debug=True)
