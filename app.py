import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.logger import get_logger
from alibi_detect.cd import KSDrift
from src.feature_store import RedisFeatureStore
from sklearn.preprocessing import StandardScaler
from prometheus_client import start_http_server, Counter, Gauge

logger = get_logger(__name__)

app = Flask(__name__, template_folder="templates")

# # Prometheus metrics
prediction_count = Counter('prediction_count', "Number of prediction count")
drift_count = Counter('drift_count', "Number of times data drift is detected")

# Load the model
MODEL_PATH = "artifacts/model/random_forest_model.pkl"
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Feature names for the new dataset
# FEATURE_NAMES = "Country,Year,Status,Life expectancy ,Adult Mortality,infant deaths,Alcohol,percentage expenditure,Hepatitis B,Measles , BMI ,under-five deaths ,Polio,Total expenditure,Diphtheria , HIV/AIDS,GDP,Population, thinness  1-19 years, thinness 5-9 years,Income composition of resources,Schooling".split(",")
FEATURE_NAMES = ['Year', 'Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling', 'Country_Albania', 'Country_Algeria', 'Country_Angola', 'Country_Antigua and Barbuda', 'Country_Argentina', 'Country_Armenia', 'Country_Australia', 'Country_Austria', 'Country_Azerbaijan', 'Country_Bahamas', 'Country_Bahrain', 'Country_Bangladesh', 'Country_Barbados', 'Country_Belarus', 'Country_Belgium', 'Country_Belize', 'Country_Benin', 'Country_Bhutan', 'Country_Bolivia (Plurinational State of)', 'Country_Bosnia and Herzegovina', 'Country_Botswana', 'Country_Brazil', 'Country_Brunei Darussalam', 'Country_Bulgaria', 'Country_Burkina Faso', 'Country_Burundi', 'Country_Cabo Verde', 'Country_Cambodia', 'Country_Cameroon', 'Country_Canada', 'Country_Central African Republic', 'Country_Chad', 'Country_Chile', 'Country_China', 'Country_Colombia', 'Country_Comoros', 'Country_Congo', 'Country_Cook Islands', 'Country_Costa Rica', 'Country_Croatia', 'Country_Cuba', 'Country_Cyprus', 'Country_Czechia', "Country_Côte d'Ivoire", "Country_Democratic People's Republic of Korea", 'Country_Democratic Republic of the Congo', 'Country_Denmark', 'Country_Djibouti', 'Country_Dominica', 'Country_Dominican Republic', 'Country_Ecuador', 'Country_Egypt', 'Country_El Salvador', 'Country_Equatorial Guinea', 'Country_Eritrea', 'Country_Estonia', 'Country_Ethiopia', 'Country_Fiji', 'Country_Finland', 'Country_France', 'Country_Gabon', 'Country_Gambia', 'Country_Georgia', 'Country_Germany', 'Country_Ghana', 'Country_Greece', 'Country_Grenada', 'Country_Guatemala', 'Country_Guinea', 'Country_Guinea-Bissau', 'Country_Guyana', 'Country_Haiti', 'Country_Honduras', 'Country_Hungary', 'Country_Iceland', 'Country_India', 'Country_Indonesia', 'Country_Iran (Islamic Republic of)', 'Country_Iraq', 'Country_Ireland', 'Country_Israel', 'Country_Italy', 'Country_Jamaica', 'Country_Japan', 'Country_Jordan', 'Country_Kazakhstan', 'Country_Kenya', 'Country_Kiribati', 'Country_Kuwait', 'Country_Kyrgyzstan', "Country_Lao People's Democratic Republic", 'Country_Latvia', 'Country_Lebanon', 'Country_Lesotho', 'Country_Liberia', 'Country_Libya', 'Country_Lithuania', 'Country_Luxembourg', 'Country_Madagascar', 'Country_Malawi', 'Country_Malaysia', 'Country_Maldives', 'Country_Mali', 'Country_Malta', 'Country_Mauritania', 'Country_Mauritius', 'Country_Mexico', 'Country_Micronesia (Federated States of)', 'Country_Monaco', 'Country_Mongolia', 'Country_Montenegro', 'Country_Morocco', 'Country_Mozambique', 'Country_Myanmar', 'Country_Namibia', 'Country_Nauru', 'Country_Nepal', 'Country_Netherlands', 'Country_New Zealand', 'Country_Nicaragua', 'Country_Niger', 'Country_Nigeria', 'Country_Niue', 'Country_Norway', 'Country_Oman', 'Country_Pakistan', 'Country_Palau', 'Country_Panama', 'Country_Papua New Guinea', 'Country_Paraguay', 'Country_Peru', 'Country_Philippines', 'Country_Poland', 'Country_Portugal', 'Country_Qatar', 'Country_Republic of Korea', 'Country_Republic of Moldova', 'Country_Romania', 'Country_Russian Federation', 'Country_Rwanda', 'Country_Saint Kitts and Nevis', 'Country_Saint Lucia', 'Country_Saint Vincent and the Grenadines', 'Country_Samoa', 'Country_San Marino', 'Country_Sao Tome and Principe', 'Country_Saudi Arabia', 'Country_Senegal', 'Country_Serbia', 'Country_Seychelles', 'Country_Sierra Leone', 'Country_Singapore', 'Country_Slovakia', 'Country_Slovenia', 'Country_Solomon Islands', 'Country_Somalia', 'Country_South Africa', 'Country_South Sudan', 'Country_Spain', 'Country_Sri Lanka', 'Country_Sudan', 'Country_Suriname', 'Country_Swaziland', 'Country_Sweden', 'Country_Switzerland', 'Country_Syrian Arab Republic', 'Country_Tajikistan', 'Country_Thailand', 'Country_The former Yugoslav republic of Macedonia', 'Country_Timor-Leste', 'Country_Togo', 'Country_Tonga', 'Country_Trinidad and Tobago', 'Country_Tunisia', 'Country_Turkey', 'Country_Turkmenistan', 'Country_Uganda', 'Country_Ukraine', 'Country_United Arab Emirates', 'Country_United Kingdom of Great Britain and Northern Ireland', 'Country_United Republic of Tanzania', 'Country_United States of America', 'Country_Uruguay', 'Country_Uzbekistan', 'Country_Vanuatu', 'Country_Venezuela (Bolivarian Republic of)', 'Country_Viet Nam', 'Country_Yemen', 'Country_Zambia', 'Country_Zimbabwe', 'Status_Developing']

feature_store = RedisFeatureStore()
scaler = StandardScaler()

# Fit the scaler on reference data
def fit_scaler_on_ref_data():
    entity_ids = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(entity_ids)

    #### 
    print(list(pd.DataFrame.from_dict(all_features, orient='index').columns))
    all_features_df = pd.DataFrame.from_dict(all_features, orient='index')[FEATURE_NAMES]

    scaler.fit(all_features_df)
    return scaler.transform(all_features_df)

historical_data = fit_scaler_on_ref_data()
ksd = KSDrift(x_ref=historical_data, p_val=0.05)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Extract input features from the form
        Country = data["Country"]  # Assuming Country is a string
        Year = int(data["Year"])
        Status = int(data["Status"])
        Adult_Mortality = float(data["Adult_Mortality"])
        Infant_Deaths = int(data["Infant_Deaths"])
        Alcohol = float(data["Alcohol"])
        Percentage_Expenditure = float(data["Percentage_Expenditure"])
        Hepatitis_B = float(data["Hepatitis_B"])
        BMI = float(data["BMI"])
        Polio = float(data["Polio"])
        GDP = float(data["GDP"])
        Schooling = float(data["Schooling"])

        # Create a DataFrame for the input features
        features = pd.DataFrame([[
            Country, Year, Status, Adult_Mortality, Infant_Deaths, Alcohol,
            Percentage_Expenditure, Hepatitis_B, BMI, Polio, GDP, Schooling
        ]], columns=FEATURE_NAMES)

        ##### Data Drift Detection
        features_scaled = scaler.transform(features.drop(columns=['Country']))  # Exclude non-numeric features

        drift = ksd.predict(features_scaled)
        logger.info(f"Drift Response: {drift}")

        drift_response = drift.get('data', {})
        is_drift = drift_response.get('is_drift', None)

        if is_drift is not None and is_drift == 1:
            logger.info("Drift Detected....")
            drift_count.inc()

        # Make a prediction
        prediction = model.predict(features.drop(columns=['Country']))[0]
        prediction_count.inc()

        result = f"Predicted Life Expectancy: {prediction:.2f}"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)})

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response

    return Response(generate_latest(), content_type='text/plain')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)