import streamlit as st
import pandas as pd
import joblib

# Load model, label encoder and model columns once on app start
model = joblib.load('skin_disease_model.joblib')
le = joblib.load('label_encoder.joblib')
model_columns = joblib.load('model_columns.joblib')

def preprocess_user_input(user_input):
    age = user_input['age']
    age_group = 1 if 13 <= age <= 35 else 0

    weather_risk = 1 if user_input['weather'] in ['Humid', 'Rainy/Flooded'] else 0
    sun_exposure_risk = 1 if user_input['sun_exposure'] in ['Noon', 'Afternoon'] else 0
    recent_travel_flood_risk = user_input.get('recent_travel_or_flood', 0)
    drinking_water_risk = 1 if user_input['drinking_water_source'] == 'Tap' else 0
    diet_dairy_risk = 1 if user_input['diet_dairy_intake'] == 'High' else 0
    diet_sugar_risk = user_input.get('diet_high_sugar', 0)
    diet_oily_spicy_risk = 1 if user_input['diet_oily_spicy_frequency'] == 'Often' else 0
    hygiene_risk = 1 if user_input['hygiene_level'] in ['Poor', 'Average'] else 0
    hydration_risk = 1 if user_input['hydration_glasses_per_day'] < 5 else 0

    features = {
        'age_group': age_group,
        'itching': user_input.get('itching', 0),
        'skin_rash': user_input.get('skin_rash', 0),
        'nodal_skin_eruptions': user_input.get('nodal_skin_eruptions', 0),
        'burning_sensation': user_input.get('burning_sensation', 0),
        'swelling': user_input.get('swelling', 0),
        'redness': user_input.get('redness', 0),
        'pus_presence': user_input.get('pus_presence', 0),
        'scaly_skin': user_input.get('scaly_skin', 0),  # fixed here
        'blisters': user_input.get('blisters', 0),
        'pain': user_input.get('pain', 0),

        'weather': user_input['weather'],
        'sun_exposure': user_input['sun_exposure'],
        'recent_travel_or_flood': recent_travel_flood_risk,
        'drinking_water_source': user_input['drinking_water_source'],
        'bathing_water_source': user_input.get('bathing_water_source', 'Tap'),

        'diet_dairy_intake': user_input['diet_dairy_intake'],
        'diet_high_sugar': diet_sugar_risk,
        'diet_oily_spicy_frequency': user_input['diet_oily_spicy_frequency'],

        'hygiene_level': user_input['hygiene_level'],
        'hydration_glasses_per_day': user_input['hydration_glasses_per_day'],

        'weather_risk': weather_risk,
        'sun_exposure_risk': sun_exposure_risk,
        'recent_travel_flood_risk': recent_travel_flood_risk,
        'drinking_water_risk': drinking_water_risk,
        'diet_dairy_risk': diet_dairy_risk,
        'diet_sugar_risk': diet_sugar_risk,
        'diet_oily_spicy_risk': diet_oily_spicy_risk,
        'hygiene_risk': hygiene_risk,
        'hydration_risk': hydration_risk
    }

    input_df = pd.DataFrame([features])

    # One-hot encode categorical columns (same as training)
    cat_cols = ['weather', 'sun_exposure', 'drinking_water_source', 'bathing_water_source',
                'diet_dairy_intake', 'diet_oily_spicy_frequency', 'hygiene_level']

    X_encoded = pd.get_dummies(input_df, columns=cat_cols)

    # Fix column mismatch by adding missing columns as zero
    for col in model_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    X_encoded = X_encoded[model_columns]

    return X_encoded


def main():
    st.title("Skin Disease Prediction")

    age = st.number_input('Age', min_value=13, max_value=80, value=22)
    itching = st.checkbox('Itching')
    skin_rash = st.checkbox('Skin Rash')
    nodal_skin_eruptions = st.checkbox('Nodal Skin Eruptions')
    burning_sensation = st.checkbox('Burning Sensation')
    swelling = st.checkbox('Swelling')
    redness = st.checkbox('Redness')
    pus_presence = st.checkbox('Pus Presence')
    scaly_skin = st.checkbox('Scaly Skin')
    blisters = st.checkbox('Blisters')
    pain = st.checkbox('Pain')

    weather = st.selectbox('Weather', ['Hot/Dry', 'Humid', 'Rainy/Flooded', 'Cold'])
    sun_exposure = st.selectbox('Sun Exposure', ['Morning', 'Noon', 'Afternoon', 'None'])
    recent_travel_or_flood = st.selectbox('Recent Travel or Flood', [0, 1])
    drinking_water_source = st.selectbox('Drinking Water Source', ['Tap', 'Filtered (RO)', 'Boiled', 'Bottled'])
    bathing_water_source = st.selectbox('Bathing Water Source', ['Tap', 'Filtered (RO)', 'Boiled', 'Bottled'])

    diet_dairy_intake = st.selectbox('Diet Dairy Intake', ['High', 'Medium', 'Low'])
    diet_high_sugar = st.selectbox('Diet High Sugar', [0, 1])
    diet_oily_spicy_frequency = st.selectbox('Diet Oily Spicy Frequency', ['Never', 'Occasionally', 'Often'])

    hygiene_level = st.selectbox('Hygiene Level', ['Good', 'Average', 'Poor'])
    hydration_glasses_per_day = st.number_input('Hydration Glasses Per Day', min_value=1, max_value=15, value=7)

    user_input = {
        'age': age,
        'itching': int(itching),
        'skin_rash': int(skin_rash),
        'nodal_skin_eruptions': int(nodal_skin_eruptions),
        'burning_sensation': int(burning_sensation),
        'swelling': int(swelling),
        'redness': int(redness),
        'pus_presence': int(pus_presence),
        'scaly_skin': int(scaly_skin),
        'blisters': int(blisters),
        'pain': int(pain),

        'weather': weather,
        'sun_exposure': sun_exposure,
        'recent_travel_or_flood': recent_travel_or_flood,
        'drinking_water_source': drinking_water_source,
        'bathing_water_source': bathing_water_source,

        'diet_dairy_intake': diet_dairy_intake,
        'diet_high_sugar': diet_high_sugar,
        'diet_oily_spicy_frequency': diet_oily_spicy_frequency,

        'hygiene_level': hygiene_level,
        'hydration_glasses_per_day': hydration_glasses_per_day
    }

    if st.button("Predict Disease"):
        processed_input = preprocess_user_input(user_input)
        pred_encoded = model.predict(processed_input)[0]
        prediction = le.inverse_transform([pred_encoded])[0]
        st.success(f"Predicted disease: {prediction}")

if __name__ == "__main__":
    main()
