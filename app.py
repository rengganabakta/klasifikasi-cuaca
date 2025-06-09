from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from bson import json_util
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB Atlas connection
try:
    # Get MongoDB URI directly from environment variable
    mongodb_uri = os.getenv('MONGODB_URI')
    mongodb_database = os.getenv('MONGODB_DATABASE', 'sensor_db')
    mongodb_collection = os.getenv('MONGODB_COLLECTION', 'sensor_data')
    
    print("Attempting to connect to MongoDB...")
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[mongodb_database]
    sensor_collection = db[mongodb_collection]
    
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None
    db = None
    sensor_collection = None

# Load the model
try:
    with open('model.pkl', 'rb') as file:
        model_dict = pickle.load(file)
    print("Model loaded successfully!")
    print(f"Type of loaded model: {type(model_dict)}")
    print(f"Model keys: {model_dict.keys()}")
    
    # Extract components
    model = model_dict['model']
    label_encoder = model_dict['label_encoder']
    feature_names = model_dict['feature_names']
    
    print("Model components loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoder = None
    feature_names = None

def get_latest_sensor_data():
    """Get the latest sensor data from MongoDB"""
    if sensor_collection is not None:
        latest_data = sensor_collection.find_one(sort=[('timestamp', -1)])
        if latest_data:
            return {
                'temperature': latest_data.get('temperature', '-'),
                'humidity': latest_data.get('humidity', '-'),
                'pressure': latest_data.get('pressure', '-'),
                'prediction': latest_data.get('prediction', '-')
            }
    return {
        'temperature': '-',
        'humidity': '-',
        'pressure': '-',
        'prediction': '-'
    }

@app.route('/')
def home():
    # Get latest 10 sensor readings
    latest_readings = []
    if sensor_collection is not None:
        latest_readings = list(sensor_collection.find().sort('timestamp', -1).limit(10))
        # Convert ObjectId to string for JSON serialization
        latest_readings = json.loads(json_util.dumps(latest_readings))
    print(f"Latest readings fetched for home page: {latest_readings}")
    return render_template('index.html', latest_readings=latest_readings)

@app.route('/history')
def history():
    # Get all sensor readings
    all_readings = []
    if sensor_collection is not None:
        all_readings = list(sensor_collection.find().sort('timestamp', -1))
        # Convert ObjectId to string for JSON serialization
        all_readings = json.loads(json_util.dumps(all_readings))
    return render_template('history.html', readings=all_readings)

@app.route('/sensor-data', methods=['POST'])
def receive_sensor_data():
    try:
        # Get data from ESP32
        data = request.get_json()
        
        # Extract features
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        pressure = float(data.get('pressure'))
        
        # Make prediction using the model components
        if all(x is not None for x in [model, label_encoder, feature_names]):
            try:
                # Create feature array in the correct order
                features = np.array([[temperature, humidity, pressure]])
                
                # Make prediction
                prediction = model.predict(features)
                
                # Decode prediction if needed
                if hasattr(label_encoder, 'inverse_transform'):
                    prediction = label_encoder.inverse_transform(prediction)
                
                prediction_value = str(prediction[0])
                
                # Store data in MongoDB
                if sensor_collection is not None:
                    sensor_data = {
                        'temperature': temperature,
                        'humidity': humidity,
                        'pressure': pressure,
                        'prediction': prediction_value,
                        'timestamp': datetime.utcnow()
                    }
                    sensor_collection.insert_one(sensor_data)
                    print(f"Data stored in MongoDB: {sensor_data}")
                
                return jsonify({
                    'status': 'success',
                    'temperature': temperature,
                    'humidity': humidity,
                    'pressure': pressure,
                    'prediction': prediction_value
                })
            except Exception as pred_error:
                print(f"Error making prediction: {pred_error}")
                return jsonify({
                    'status': 'error',
                    'message': f'Error making prediction: {str(pred_error)}'
                }), 500
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model components not loaded properly'
            }), 500
            
    except Exception as e:
        print(f"Error in sensor data endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/latest-data', methods=['GET'])
def get_latest_data():
    """Endpoint to get the latest sensor data"""
    return jsonify(get_latest_sensor_data())

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'label_encoder_loaded': label_encoder is not None,
        'feature_names_loaded': feature_names is not None,
        'mongodb_connected': client is not None,
        'mongodb_database': mongodb_database,
        'mongodb_collection': mongodb_collection
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 