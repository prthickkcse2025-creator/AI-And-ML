from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --- 1. Train the model based on your Rules ---
df = pd.read_csv('Height_Weight.csv')
X = df[['Height']] 
y = df['Weight']

# Splitting data as per Linear_Regression_Rules.md
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- 2. HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Weight Predictor</title>
    <style>
        body { font-family: sans-serif; display: flex; justify-content: center; margin-top: 50px; background-color: #f4f4f9; }
        .card { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 300px; }
        input { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
        button { width: 100%; padding: 10px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        #result { margin-top: 20px; font-weight: bold; text-align: center; color: #333; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Weight Predictor</h2>
        <p>Enter Height (cm):</p>
        <input type="number" id="heightInput" placeholder="e.g. 175">
        <button onclick="getPrediction()">Predict Weight</button>
        <div id="result"></div>
    </div>

    <script>
        async function getPrediction() {
            const height = document.getElementById('heightInput').value;
            const resultDiv = document.getElementById('result');
            
            if(!height) {
                resultDiv.innerText = "Please enter a value";
                return;
            }

            resultDiv.innerText = "Calculating...";

            try {
                const response = await fetch(`/predict?height=${height}`);
                const data = await response.json();
                
                if(data.predicted_weight) {
                    resultDiv.innerText = `Predicted Weight: ${data.predicted_weight} kg`;
                } else {
                    resultDiv.innerText = "Error getting prediction";
                }
            } catch (error) {
                resultDiv.innerText = "Server Error";
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["GET"])
def predict():
    height_input = request.args.get('height')
    try:
        height_float = float(height_input)
        # Scale the single input point using the scaler fitted on training data
        scaled_height = scaler.transform([[height_float]])
        prediction = model.predict(scaled_height)
        
        return jsonify({
            "input_height": height_float,
            "predicted_weight": round(prediction[0], 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)