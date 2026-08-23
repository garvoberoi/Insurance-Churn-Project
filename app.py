from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and SHAP explainer
xgb_model = joblib.load('xgb_model.pkl')
explainer = joblib.load('shap_explainer.pkl')

# Retrieving Column structure
model_columns = pd.read_csv('X_train.csv', nrows=0).columns.tolist()

print('Model loaded successfully')
print(f'Expecting {len(model_columns)} features per customer')

@app.route('/predict', methods=['POST'])
def predict():
    # JSON from request
    customer_data = request.get_json()

    # Convert it into a single-row DataFrame
    df = pd.DataFrame([customer_data])

    # One-hot encoding
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # verifying column names etc to match both source and target
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Run the prediction
    churn_probability = xgb_model.predict_proba(df_final)[0][1]

    # Get SHAP explanation
    shap_values = explainer.shap_values(df_final)
    reasons_df = pd.DataFrame({
        'feature': model_columns,
        'shap_value': shap_values[0]
    }).sort_values('shap_value', key=abs, ascending=False).head(3)

    # Build the response
    response = {
        'churn_probability': round(float(churn_probability), 3),
        'risk_band': 'High' if churn_probability > 0.6 else 'Medium' if churn_probability > 0.3 else 'Low',
        'top_reasons': [
            {'feature': row['feature'], 'impact': round(float(row['shap_value']), 3)}
            for _, row in reasons_df.iterrows()
        ]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)