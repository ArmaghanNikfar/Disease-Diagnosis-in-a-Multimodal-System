from flask import Flask, request, jsonify
import re
from fastapi import File, UploadFile
from flask_cors import CORS
from transformers import pipeline
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import easyocr
import cv2
from textblob import TextBlob

app = Flask(__name__)
CORS(app)
required_features = [
    'Body Temp (°C)', 'Cough Type', 'Sore Throat', 'Runny Nose', 'Fatigue',
    'Loss of Smell/Taste', 'Shortness of Breath', 'Headache', 'Muscle Aches', 
    'Skin',  
    'Pulse Oximetry', 'Count Of Breath Per Minute', 'Heart Rate', 'Rash', 
    'Edema', 'Stomach Pain', 'Diarrhea', 'Vomiting', 'Pain in Urination', 
    'Wheezing', 'Dyspnea'
]
model = load_model('lstm_gpc_model.h5')
label_encoders = {
    'Cough Type': LabelEncoder().fit(['no', 'mild', 'wet', 'dry']),
    'Sore Throat': LabelEncoder().fit(['no', 'yes']),
    'Runny Nose': LabelEncoder().fit(['no', 'yes', 'sometimes']),
    'Fatigue': LabelEncoder().fit(['no', 'mild', 'severe']),
    'Loss of Smell/Taste': LabelEncoder().fit(['no', 'yes']),
    'Shortness of Breath': LabelEncoder().fit(['no', 'yes']),
    'Headache': LabelEncoder().fit(['no', 'yes']),
    'Muscle Aches': LabelEncoder().fit(['no', 'sometimes', 'severe']),
    'Skin': LabelEncoder().fit(['dry', 'wet', 'normal']),
    'Rash': LabelEncoder().fit(['yes', 'no']),
    'Edema': LabelEncoder().fit(['yes', 'no']),
    'Stomach Pain': LabelEncoder().fit(['yes', 'no']),
    'Diarrhea': LabelEncoder().fit(['yes', 'no']),
    'Vomiting': LabelEncoder().fit(['yes', 'no']),
    'Pain in Urination': LabelEncoder().fit(['yes', 'no']),
    'Wheezing': LabelEncoder().fit(['yes', 'no']),
    'Dyspnea': LabelEncoder().fit(['yes','no']),
    'Diagnosis': LabelEncoder().fit(['covid-19', 'cold', 'flu', 'healthy'])
}

questions = {
    'Body Temp (°C)': "What is your body temperature in Celsius?",
    'Cough Type': "What type of cough do you have? (No, Mild, Wet, Dry)",
    'Sore Throat': "Do you have a sore throat? (Yes or No)",
    'Runny Nose': "Do you have a runny nose? (No, Yes, Sometimes)",
    'Fatigue': "How would you describe your fatigue? (No, Mild, Severe)",
    'Loss of Smell/Taste': "Have you lost your sense of smell or taste? (Yes or No)",
    'Shortness of Breath': "Do you experience shortness of breath? (Yes or No)",
    'Headache': "Do you have a headache? (Yes or No)",
    'Muscle Aches': "Do you have muscle aches? (No, Sometimes, Severe)",
    'Skin': "How would you describe your skin? (Dry, Wet, Normal)",
    'Pulse Oximetry': "What is your oxygen saturation level (Pulse Oximetry percentage, e.g 0.92)?",
    'Count Of Breath Per Minute': "How many breaths do you take per minute?",
    'Heart Rate': "What is your heart rate (beats per minute)?",
    'Rash': "Do you have any rashes? (Yes or No)",
    'Edema': "Do you have edema (swelling)? (Yes or No)",
    'Stomach Pain': "Do you have stomach pain? (Yes or No)",
    'Diarrhea': "Do you have diarrhea? (Yes or No)",
    'Vomiting': "Have you experienced vomiting? (Yes or No)",
    'Pain in Urination': "Do you experience pain during urination? (Yes or No)",
    'Wheezing': "Do you have wheezing? (Yes or No)",
    'Dyspnea': "What type of dyspnea do you experience? (Yes or No)"
}
regex_patterns = {
    'Body Temp (°C)': r'\b(\d+\.?\d*)\b',  
    'Cough Type': r'\b(no|mild|wet|dry)\b',  
    'Sore Throat': r'\b(yes|no)\b',
    'Runny Nose': r'\b(no|yes|sometimes)\b',
    'Fatigue': r'\b(no|mild|severe)\b',
    'Loss of Smell/Taste': r'\b(yes|no)\b',
    'Shortness of Breath': r'\b(yes|no)\b',
    'Headache': r'\b(yes|no)\b',
    'Muscle Aches': r'\b(no|sometimes|severe)\b',
    'Pulse Oximetry': r'\b(\d+)\b', 
    'Count Of Breath Per Minute': r'\b(\d+)\b',
    'Heart Rate': r'\b(\d+)\b',
    'Skin': r'\b(dry|wet|normal)\b',
    'Rash': r'\b(yes|no)\b',
    'Edema': r'\b(yes|no)\b',
    'Stomach Pain': r'\b(yes|no)\b',
    'Diarrhea': r'\b(yes|no)\b',
    'Vomiting': r'\b(yes|no)\b',
    'Pain in Urination': r'\b(yes|no)\b',
    'Wheezing': r'\b(yes|no)\b',
    'Dyspnea': r'\b(yes|no)\b'
}
user_data = {}
def tokenize_and_normalize(text):
    """
    1. converts text to lowercase.
    2. splits the text into individual words.
    """
    text = text.lower()  
    tokens = re.findall(r'\b\w+\b', text)  
    return tokens


def extract_feature_value(feature, answer):
  
    tokens = tokenize_and_normalize(answer) 
  
    if feature in regex_patterns:
        for token in tokens:
            blob = TextBlob(token)
            correct_answers = str(blob.correct())
            print(correct_answers)
            if re.fullmatch(regex_patterns[feature], correct_answers):
                if feature in label_encoders:
                    valid_values = label_encoders[feature].classes_
                    if correct_answers not in valid_values:
                        print(f"'{correct_answers}' is not a valid value for {feature}. Valid values: {valid_values}")
                        continue
                    
                return correct_answers
    return None

@app.route('/collect_data', methods=['POST'])
def collect_data():
    global user_data
    
    data = request.get_json()
    user_data = data.get("user_data", {})

    missing_features = [feature for feature in required_features if feature not in user_data]
    
    if set(user_data.keys()) == required_features:
        return jsonify({"message": "All features are complete.","nextLevel": True})
    
    missing_questions = {feature: questions[feature] for feature in missing_features}
    
    return jsonify({"missing_questions": missing_questions})


@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    global user_data
    
    data = request.get_json()
    feature = data.get("feature")
    answer = data.get("answer")
    
    if feature not in required_features:
        return jsonify({"error": "Invalid feature"}), 400
    
    valid_answer = extract_feature_value(feature, answer)
    
    if valid_answer:
        user_data[feature] = valid_answer
        if len(user_data) == len(required_features): return jsonify({"message": f"Data for {feature} saved successfully.",
                                                                      "user_data": len(user_data),
                                                                        "excepted": len(required_features),
                                                                        "userDataList": user_data,
                                                                        "isComplated":'True'})
        else:return jsonify({"message": "All features are complete.", "nextLevel": True})
    else:
        return jsonify({"error": "Invalid answer format"}), 400
    

@app.route("/extract_lab_features/", methods=["POST"])
def extract_lab_features():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    contents = file.read()
    
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    reader = easyocr.Reader(['ru'])
    results = reader.readtext(image)
    extracted_texts = [text.strip() for (_, text, _) in results]

    lab_features = {
        'Эпителий плоский': '0',
        'Лейкоциты': '0',
        'С-реактивный белок (СРБ)': '0',
        'Общий Белок': '0',
        'Тромбоциты': '0',
        'Лимфоциты': '0'
    }

    for i, text in enumerate(extracted_texts):
        for feature in lab_features.keys():
            if feature in text:
                if i + 1 < len(extracted_texts):  
                    next_value = extracted_texts[i + 1]
                    match = re.search(r"\d+[\.,]?\d*", next_value)
                    if match:
                        lab_features[feature] = match.group()
    
    return jsonify({"recognition": lab_features, "isCompleted": True})

@app.route('/predict', methods=['POST'])
def predict_diagnosis_nlp():
    data = request.get_json()
    user_data = data.get("user_data", {})
    lab_features_list = data.get("lab_features", []) 
    lab_features = lab_features_list[0] if lab_features_list else {}

    if not isinstance(lab_features, dict):
        lab_features = {}
    if not isinstance(user_data, dict):
        user_data = {}

    for feature, value in user_data.items():
        if feature in regex_patterns:
            if isinstance(value, str):
                match = re.search(regex_patterns[feature], value.lower())
                if match:
                    valid_value = match.group(1) if match.groups() else match.group(0)
                    if feature in label_encoders:
                        if valid_value in label_encoders[feature].classes_:
                            user_data[feature] = valid_value
                        else:
                            user_data[feature] = None  
                    else:
                        user_data[feature] = valid_value
                else:
                    user_data[feature] = None  
            else:
                try:
                    user_data[feature] = float(value)
                except ValueError:
                    user_data[feature] = 0.0  
        else:

            pass

    numeric_keys = [
        'Body Temp (°C)', 'Count Of Breath Per Minute', 'Heart Rate', 'Pulse Oximetry',
        'Лейкоциты', 'Лимфоциты', 'Общий Белок', 'С-реактивный белок (СРБ)', 'Тромбоциты', 'Эпителий плоский'
    ]


    for key, value in user_data.items():
        if key in numeric_keys and isinstance(value, str):
            try:
                user_data[key] = float(value)
            except ValueError:
                user_data[key] = 0.0 

    for key, value in lab_features.items():
        if key in numeric_keys and isinstance(value, str):
            try:
                lab_features[key] = float(value)
            except ValueError:
                lab_features[key] = 0.0 


    combined_data = {**lab_features, **user_data}
    print(combined_data)
    if not user_data:
        return jsonify({"error": "No user data provided"}), 400

    for feature in label_encoders:
        if feature in combined_data:
            if isinstance(combined_data[feature], str) and combined_data[feature] in label_encoders[feature].classes_:
                combined_data[feature] = label_encoders[feature].transform([combined_data[feature]])[0]

    input_features = np.array(list(combined_data.values())).reshape((1, 1, len(combined_data)))
    input_features = (input_features - np.min(input_features)) / (np.max(input_features) - np.min(input_features) + 1e-6)

    prediction = model.predict(input_features)
    predicted_class = np.argmax(prediction[0])
    class_name = label_encoders['Diagnosis'].classes_[predicted_class]

    return jsonify({"PredictedClass": class_name, "Probabilities": prediction[0].tolist()})
if __name__ == '__main__':
    app.run(debug=True)
