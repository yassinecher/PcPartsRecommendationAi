import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for the app, allowing requests from specific origins
CORS(app, resources={r"/recommend": {"origins": ["http://localhost:3000"]}})

# Get the absolute path to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load models and data
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    build_table = pd.read_csv(os.path.join(MODELS_DIR, 'build_table.csv'))
    feature_cols = [
        'CPU_Core_Count', 'CPU_Performance_Core_Clock',
        'Memory_Speed', 'Video Card_Memory', 'Power Supply_Wattage'
    ]
    X = build_table[feature_cols].astype(float)
    X_scaled = scaler.transform(X)
    knn_model = joblib.load(os.path.join(MODELS_DIR, 'knn_model.pkl'))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    logger.info("Models and data loaded successfully")
except FileNotFoundError as e:
    logger.error(f"FileNotFoundError: {e}. Ensure files exist in {MODELS_DIR}")
    exit(1)
except Exception as e:
    logger.error(f"Error loading models or data: {e}")
    exit(1)

@app.route('/recommend', methods=['POST'])
def recommend_builds():
    try:
        # Get user input from JSON request
        user_input = request.get_json()
        if not user_input or not isinstance(user_input, dict):
            return jsonify({'error': 'Invalid input. Provide a JSON object with feature values.'}), 400

        # Validate input features
        invalid_features = [key for key in user_input if key not in feature_cols]
        if invalid_features:
            return jsonify({'error': f'Invalid features: {invalid_features}. Valid features: {feature_cols}'}), 400

        # Complete user_input with default values if missing
        user_input_complete = {}
        for col in feature_cols:
            if col in user_input:
                try:
                    user_input_complete[col] = float(user_input[col])
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid value for {col}: {user_input[col]}. Must be a number.'}), 400
            else:
                user_input_complete[col] = build_table[col].median()

        # Create user DataFrame
        user_vector = np.array([list(user_input_complete.values())]).astype(float)
        user_df = pd.DataFrame(user_vector, columns=feature_cols)
        user_scaled = scaler.transform(user_df)

        # --- 1. KNN ---
        _, indices_knn = knn_model.named_steps['nearestneighbors'].kneighbors(user_scaled)
        recs_knn = build_table.iloc[indices_knn[0]].head(5)

        # Prepare response
        recommendations = {
            'knn': recs_knn[['Build Title'] + feature_cols].to_dict(orient='records'),
        }

        return jsonify({
            'message': f'Recommendations generated for {list(user_input.keys())} = {user_input}',
            'recommendations': recommendations
        }), 200

    except Exception as e:
        logger.error(f"Error in recommend_builds: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)