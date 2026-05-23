import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from flask import Flask, request, jsonify, render_template_string
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_names = None

    def load_data(self):
        """Load and display dataset information"""
        print("🚗Loading Car Price Prediction Dataset...")
        self.df = pd.read_csv('car_price_prediction.csv')
        print(f"✅Dataset loaded successfully!")
        print(f"📊 Shape: {self.df.shape}")
        return self.df

    def preprocess_data(self):
        """Data cleaning and preprocessing"""
        print("\n🧹 Data Preprocessing:")
        # Remove ID column
        if 'ID' in self.df.columns:
            self.df = self.df.drop('ID', axis=1)
        
        # Handle missing values
        self.df = self.df.dropna(subset=['Price'])
        
        # Outlier Handling for Price
        # Keeping prices between 5th and 95th percentile to remove extremes
        q_low = self.df["Price"].quantile(0.05)
        q_hi  = self.df["Price"].quantile(0.95)
        self.df = self.df[(self.df["Price"] < q_hi) & (self.df["Price"] > q_low)]
        print(f"📉 After outlier removal: {self.df.shape[0]} samples")

        # Clean Levy column
        self.df['Levy'] = self.df['Levy'].replace('-', '0')
        self.df['Levy'] = pd.to_numeric(self.df['Levy'], errors='coerce')
        self.df['Levy'] = self.df['Levy'].fillna(0)
        
        # Clean Engine volume
        self.df['Engine volume'] = self.df['Engine volume'].astype(str).str.replace(' Turbo', '')
        self.df['Engine volume'] = pd.to_numeric(self.df['Engine volume'], errors='coerce')
        self.df['Engine volume'] = self.df['Engine volume'].fillna(self.df['Engine volume'].median())
        
        # Clean Mileage
        self.df['Mileage'] = self.df['Mileage'].astype(str).str.replace(' km', '').str.replace(',', '')
        self.df['Mileage'] = pd.to_numeric(self.df['Mileage'], errors='coerce')
        self.df['Mileage'] = self.df['Mileage'].fillna(self.df['Mileage'].median())
        
        # Clean Cylinders
        self.df['Cylinders'] = self.df['Cylinders'].fillna(self.df['Cylinders'].median())
        
        # Create car age feature
        self.df['Car_Age'] = 2025 - self.df['Prod. year']
        # self.df = self.df.drop('Prod. year', axis=1) # Keep it or drop it, usually drop
        
        # Encode categorical variables
        categorical_cols = ['Manufacturer', 'Model', 'Category', 'Leather interior',
                          'Fuel type', 'Gear box type', 'Drive wheels', 'Doors',
                          'Wheel', 'Color']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        print("✅Data preprocessing completed!")

    def feature_engineering(self):
        """Feature engineering and selection"""
        print("\n⚙️Feature Engineering:")
        # Select features
        self.feature_names = [col for col in self.df.columns if col != 'Price']
        X = self.df[self.feature_names]
        y = self.df['Price']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        print(f"✅Features prepared: {len(self.feature_names)} features")
        print(f"✅Training set: {self.X_train.shape[0]} samples")
        print(f"✅Test set: {self.X_test.shape[0]} samples")

    def train_models(self):
        """Train Random Forest model"""
        print("\n🤖Training Random Forest Model:")
        # Use Random Forest with reasonable depth to avoid overfitting
        self.best_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        print("Training Random Forest...")
        self.best_model.fit(self.X_train, self.y_train)
        y_pred = self.best_model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        print(f" R² Score: {r2:.4f}")
        print(f" MAE: {mae:.2f}")
        print(f" RMSE: {rmse:.2f}")
        print(f"\n🏆Model trained successfully! (R² = {r2:.4f})")

    def save_model(self):
        """Save the trained model and preprocessing objects"""
        print("\n💾 Saving Model:")
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        with open('car_price_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print("✅Model saved as 'car_price_model.pkl'")

    def run_pipeline(self):
        """Run the complete ML pipeline"""
        print("🚀Starting Car Price Prediction Pipeline...")
        self.load_data()
        self.preprocess_data()
        self.feature_engineering()
        self.train_models()
        self.save_model()
        print("\n🎉Pipeline completed successfully!")

# Flask Web Application
app = Flask(__name__)

# Cache for manufacturers and models to populate dropdowns
CACHE = {
    'manufacturers': [],
    'fuel_types': [],
    'gear_boxes': [],
    'drive_wheels': []
}

def load_cache():
    if os.path.exists('car_price_prediction.csv'):
        df = pd.read_csv('car_price_prediction.csv')
        CACHE['manufacturers'] = sorted(df['Manufacturer'].unique().tolist())
        CACHE['fuel_types'] = sorted(df['Fuel type'].unique().tolist())
        CACHE['gear_boxes'] = sorted(df['Gear box type'].unique().tolist())
        CACHE['drive_wheels'] = sorted(df['Drive wheels'].unique().tolist())

def format_inr(amount):
    return f"₹{amount:,.2f}"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Price Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .form-row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .form-group {
            flex: 1;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
            box-sizing: border-box;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
            margin-top: 20px;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .result {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #667eea;
        }
        .price {
            font-size: 2.5em;
            color: #667eea;
            font-weight: bold;
            margin: 10px 0;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗Car Price Predictor</h1>
            <p>Get accurate car price predictions using advanced machine learning</p>
        </div>
        <form method="POST">
            <div class="form-row">
                <div class="form-group">
                    <label>Manufacturer:</label>
                    <select name="manufacturer" required>
                        <option value="">Select Manufacturer</option>
                        {% for m in manufacturers %}
                        <option value="{{ m }}">{{ m }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Model Name:</label>
                    <input type="text" name="model" placeholder="e.g., RX 450" required>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Production Year:</label>
                    <input type="number" name="year" min="1990" max="2025" value="2020" required>
                </div>
                <div class="form-group">
                    <label>Mileage (km):</label>
                    <input type="number" name="mileage" min="0" value="50000" required>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Fuel Type:</label>
                    <select name="fuel_type" required>
                        <option value="">Select Fuel Type</option>
                        {% for f in fuel_types %}
                        <option value="{{ f }}">{{ f }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Engine Volume:</label>
                    <input type="number" name="engine_volume" step="0.1" min="0.5" max="8.0" value="2.0" required>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Gear Box Type:</label>
                    <select name="gear_box" required>
                        <option value="">Select Gear Box</option>
                        {% for g in gear_boxes %}
                        <option value="{{ g }}">{{ g }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Drive Wheels:</label>
                    <select name="drive_wheels" required>
                        <option value="">Select Drive Wheels</option>
                        {% for d in drive_wheels %}
                        <option value="{{ d }}">{{ d }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
            <button type="submit" class="btn">🔮 Predict Price</button>
        </form>

        {% if prediction %}
        <div class="result">
            <h3>🎯Predicted Car Price</h3>
            <div class="price">{{ prediction_str }}</div>
            <p>This prediction is based on the car specifications you provided and our trained machine learning model.</p>
            <p><small>Note: Price converted to Indian Rupees (1 USD ≈ ₹83)</small></p>
        </div>
        {% endif %}

        {% if error %}
        <div class="error">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, 
                                 manufacturers=CACHE['manufacturers'],
                                 fuel_types=CACHE['fuel_types'],
                                 gear_boxes=CACHE['gear_boxes'],
                                 drive_wheels=CACHE['drive_wheels'])

@app.route('/', methods=['POST'])
def predict():
    try:
        # Get form data
        manufacturer = request.form['manufacturer']
        model_name = request.form['model']
        year = int(request.form['year'])
        mileage = int(request.form['mileage'])
        fuel_type = request.form['fuel_type']
        engine_volume = float(request.form['engine_volume'])
        gear_box = request.form['gear_box']
        drive_wheels = request.form['drive_wheels']

        # Load model
        if not os.path.exists('car_price_model.pkl'):
            return render_template_string(HTML_TEMPLATE, 
                                         manufacturers=CACHE['manufacturers'],
                                         fuel_types=CACHE['fuel_types'],
                                         gear_boxes=CACHE['gear_boxes'],
                                         drive_wheels=CACHE['drive_wheels'],
                                         error="Model file not found. Please train the model first.")

        with open('car_price_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
        feature_names = model_data['feature_names']

        # Create feature vector
        features = []
        for feature in feature_names:
            if feature == 'Manufacturer':
                val = manufacturer if manufacturer in label_encoders[feature].classes_ else label_encoders[feature].classes_[0]
                features.append(label_encoders[feature].transform([val])[0])
            elif feature == 'Model':
                # Handle unseen models by using the most common one or a default
                val = model_name if model_name in label_encoders[feature].classes_ else label_encoders[feature].classes_[0]
                features.append(label_encoders[feature].transform([val])[0])
            elif feature == 'Car_Age':
                features.append(2025 - year)
            elif feature == 'Prod. year':
                features.append(year)
            elif feature == 'Mileage':
                features.append(mileage)
            elif feature == 'Fuel type':
                val = fuel_type if fuel_type in label_encoders[feature].classes_ else label_encoders[feature].classes_[0]
                features.append(label_encoders[feature].transform([val])[0])
            elif feature == 'Engine volume':
                features.append(engine_volume)
            elif feature == 'Gear box type':
                val = gear_box if gear_box in label_encoders[feature].classes_ else label_encoders[feature].classes_[0]
                features.append(label_encoders[feature].transform([val])[0])
            elif feature == 'Drive wheels':
                val = drive_wheels if drive_wheels in label_encoders[feature].classes_ else label_encoders[feature].classes_[0]
                features.append(label_encoders[feature].transform([val])[0])
            else:
                # Default values for other features
                if feature in label_encoders:
                    # Just use the first class for default
                    features.append(label_encoders[feature].transform([label_encoders[feature].classes_[0]])[0])
                else:
                    # Numeric defaults
                    if feature == 'Cylinders': features.append(4)
                    elif feature == 'Airbags': features.append(8)
                    elif feature == 'Levy': features.append(0)
                    else: features.append(0)

        # Scale and Predict
        features_scaled = scaler.transform([features])
        prediction_usd = model.predict(features_scaled)[0]
        prediction_inr = max(0, prediction_usd * 83) # Ensure positive
        
        return render_template_string(HTML_TEMPLATE, 
                                     manufacturers=CACHE['manufacturers'],
                                     fuel_types=CACHE['fuel_types'],
                                     gear_boxes=CACHE['gear_boxes'],
                                     drive_wheels=CACHE['drive_wheels'],
                                     prediction=prediction_usd,
                                     prediction_str=format_inr(prediction_inr))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, 
                                     manufacturers=CACHE['manufacturers'],
                                     fuel_types=CACHE['fuel_types'],
                                     gear_boxes=CACHE['gear_boxes'],
                                     drive_wheels=CACHE['drive_wheels'],
                                     error=str(e))

if __name__ == '__main__':
    load_cache()
    
    # Check if model exists, if not train it
    if not os.path.exists('car_price_model.pkl'):
        print("🚀Model not found. Starting training...")
        predictor = CarPricePredictor()
        predictor.run_pipeline()
    else:
        print("✅Found existing model 'car_price_model.pkl'.")
    
    print("\n🌐 Starting Web Application...")
    app.run(debug=False, host='0.0.0.0', port=5000)
