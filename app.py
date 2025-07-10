from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# === Load Model and Scalers ===
with open("rf_model_clean.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_standard.pkl", "rb") as f:
    scaler_std = pickle.load(f)

with open("scaler_minmax.pkl", "rb") as f:
    scaler_minmax = pickle.load(f)

with open("scaler_robust.pkl", "rb") as f:
    scaler_robust = pickle.load(f)

# === Label Mapping ===
label_map = {
    0: "Cereal",
    1: "Commercial",
    2: "Fruit",
    3: "Pulse"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Apply individual scalers (same as training)
        scaled_N = scaler_std.transform([[N, 0, 0, 0]])[0][0]
        scaled_temperature = scaler_std.transform([[0, temperature, 0, 0]])[0][1]
        scaled_ph = scaler_std.transform([[0, 0, ph, 0]])[0][2]
        scaled_rainfall = scaler_std.transform([[0, 0, 0, rainfall]])[0][3]

        scaled_P = scaler_minmax.transform([[P]])[0][0]
        scaled_K = scaler_robust.transform([[K, 0]])[0][0]
        scaled_humidity = scaler_robust.transform([[0, humidity]])[0][1]

        # Final input array
        final_features = np.array([
            scaled_N,
            scaled_P,
            scaled_K,
            scaled_temperature,
            scaled_humidity,
            scaled_ph,
            scaled_rainfall
        ]).reshape(1, -1)

        # Predict
        prediction = model.predict(final_features)[0]
        crop_type = label_map.get(int(prediction), "Unknown")

        return render_template('index.html', prediction_text=f"🌾 Recommended Crop Type: <b>{crop_type}</b>")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
