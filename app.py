from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
model_file = 'traffic_density_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Load the dataset to get unique values for input options
df = pd.read_csv("futuristic_city_traffic.csv")

@app.route('/')
def home():
    return render_template('index.html', 
                           cities=df['City'].unique(),
                           vehicle_types=df['Vehicle Type'].unique(),
                           weather_conditions=df['Weather'].unique(),
                           economic_conditions=df['Economic Condition'].unique(),
                           days_of_week=df['Day Of Week'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_data = {
        'City': data['City'],
        'Vehicle Type': data['Vehicle Type'],
        'Weather': data['Weather'],
        'Economic Condition': data['Economic Condition'],
        'Day Of Week': data['Day Of Week'],
        'Hour Of Day': int(data['Hour Of Day']),
        'Speed': int(data['Speed']),
        'Energy Consumption': int(data['Energy Consumption']),
        'Is Peak Hour': int(data['Is Peak Hour']),
        'Random Event Occurred': int(data['Random Event Occurred'])
    }
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return jsonify({'Predicted Traffic Density': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
