from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('data/Train.csv')
data['date_time'] = pd.to_datetime(data['date_time'])
data = data.sort_values(by=['date_time']).reset_index(drop=True)

# Feature engineering
data['is_holiday'] = data['is_holiday'].apply(lambda x: 0 if x == 'None' else 1)
data['hour'] = data['date_time'].dt.hour
data['month_day'] = data['date_time'].dt.day
data['weekday'] = data['date_time'].dt.weekday + 1
data['month'] = data['date_time'].dt.month
data['year'] = data['date_time'].dt.year

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_labels = encoder.fit_transform(data[['weather_type', 'weather_description']])
encoded_features = encoder.get_feature_names_out(['weather_type', 'weather_description'])

# Combine features
features = data[['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month']]
features = np.hstack([features, encoded_labels])
target = data['traffic_volume']

# Scaling
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(features)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

# Train model
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)


# Flask routes
@app.route('/')
def root():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Prepare input data based on user input
    lis = []
    for i in range(0,24):
        dic = dict()
        dic["day"] = request.form['date']
        input_data = {
            'is_holiday': 1 if request.form['isholiday'] == 'yes' else 0,
            'temperature': int(request.form['temperature']),
            'weekday': 0,
            # 'hour': int(request.form['time'][:2]),
            'hour': i,
            'month_day': int(request.form['date'][8:]),
            'year': int(request.form['date'][:4]),
            'month': int(request.form['date'][5:7])
        }

        # One-hot encode user input for `x0` and `x1`
        x0_input = request.form['x0']
        x1_input = request.form['x1']
        x0_encoded = {f'weather_type_{val}': 0 for val in encoder.categories_[0]}
        x1_encoded = {f'weather_description_{val}': 0 for val in encoder.categories_[1]}
        x0_encoded[f'weather_type_{x0_input}'] = 1
        x1_encoded[f'weather_description_{x1_input}'] = 1

        # Final input for prediction
        final_input = list(input_data.values()) + list(x0_encoded.values()) + list(x1_encoded.values())
        final_input_scaled = x_scaler.transform([final_input])

        # Predict
        output = regr.predict(final_input_scaled)
        predicted_traffic = y_scaler.inverse_transform(output.reshape(-1, 1)).flatten()
        print("testing...")
    
        dic["hour"] = i
        dic["data"] = int(predicted_traffic[0])
        lis.append(dic)
    # print(predicted_traffic[0])
    return render_template('output.html', data1=input_data, prediction=lis)


if __name__ == '__main__':
    app.run(debug=True)
