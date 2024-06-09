import streamlit as st
import joblib

# Memuat model dan vectorizer
model = joblib.load('text_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI
st.title('SMS Spam Classifier')
st.write('Enter an SMS message to classify it as spam or ham.')


st.write("
This app predicts Spam or Ham Message
")

# Input teks dari pengguna
user_input = st.text_area('Enter SMS message here:')

# Tombol untuk memicu klasifikasi
if st.button('Classify'):
    # Transform input text to match the training data format
    user_input_transformed = vectorizer.transform([user_input])

    # Make prediction
    prediction = model.predict(user_input_transformed)
    prediction_label = 'Spam' if prediction[0] == 1 else 'Ham'

    st.write('Predicted class:', prediction_label)
