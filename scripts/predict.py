from flask import Flask, request, jsonify
import pickle
import pandas as pd
import xgboost as xgb

model_file = 'xgb_boost.pkl'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('transaction-fraud')

@app.route('/')
def health_check():
    return jsonify({'status': 'healthy'}), 200


@app.route('/predict', methods = ['POST'])
def predict():
    customer = request.get_json()
    customer_df = pd.DataFrame([customer])
    customer_dmatrix = xgb.DMatrix(customer_df)

    y_pred_probs = model.predict(customer_dmatrix)
    y_pred = (y_pred_probs > 0.5).astype(bool)

    result =  {
        'fraud' : bool(y_pred),
        'fraud_probability' : float(y_pred_probs) 
    }
    return jsonify(result)

    
    
if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 8080)
