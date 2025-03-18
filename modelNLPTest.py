import numpy as np
from transformers import pipeline
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import re
import easyocr
import cv2
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

required_features = [
    'Body Temp (°C)', 'Cough Type', 'Sore Throat', 'Runny Nose', 'Fatigue',
    'Loss of Smell/Taste', 'Shortness of Breath', 'Headache', 'Muscle Aches', 
    'Skin',  
    'Pulse Oximetry', 'Count Of Breath Per Minute', 'Heart Rate', 'Rash', 
    'Edema', 'Stomach Pain', 'Diarrhea', 'Vomiting', 'Pain in Urination', 
    'Wheezing', 'Dyspnea'
]

regex_patterns = {
    'Body Temp (°C)': r'(\d+\.?\d*)',  
    'Cough Type': r'(no|mild|wet|dry)',
    'Sore Throat': r'(yes|no)',
    'Runny Nose': r'(no|yes|sometimes)',
    'Fatigue': r'(no|mild|severe)',
    'Loss of Smell/Taste': r'(yes|no)',
    'Shortness of Breath': r'(yes|no)',
    'Headache': r'(yes|no)',
    'Muscle Aches': r'(no|sometimes|severe)',
    'Pulse Oximetry': r'(\d+)',  
    'Count Of Breath Per Minute': r'(\d+)',  
    'Heart Rate': r'(\d+)',  
    'Skin': r'(dry|wet|normal)',
    'Rash': r'(yes|no)',
    'Edema': r'(yes|no)',
    'Stomach Pain': r'(yes|no)',
    'Diarrhea': r'(yes|no)',
    'Vomiting': r'(yes|no)',
    'Pain in Urination': r'(yes|no)',
    'Wheezing': r'(yes|no)',
    'Dyspnea': r'(yes|no)'
}

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
            if re.fullmatch(regex_patterns[feature], token):
                if feature in label_encoders:
                    valid_values = label_encoders[feature].classes_
                    if token not in valid_values:
                        print(f"'{token}' is not a valid value for {feature}. Valid values: {valid_values}")
                        continue
                return token
    return None


def collect_data_nlp():
    user_data = {}
    print("We will ask you a series of health-related questions. Answer them as accurately as possible.")

    for feature, question in questions.items():
        while True:
            answer = input(f"{question} > ")
            extracted_value = extract_feature_value(feature, answer)
            if extracted_value:
                if feature in label_encoders:
                    user_data[feature] = label_encoders[feature].transform([extracted_value])[0]
                else:
                    user_data[feature] = float(extracted_value)
                break
            else:
                print("Invalid input. Please try again.")
    return user_data
# Function to extract lab features from the image
def extract_lab_features():
    image_path = input("Do you have a lab test image to upload? (yes/no): ").strip().lower()
    lab_features = {
        'Эпителий плоский': 0,
        'Лейкоциты': 0,
        'С-реактивный белок (СРБ)': 0,
        'Общий Белок': 0,
        'Тромбоциты': 0,
        'Лимфоциты': 0
    }
    
    if image_path == 'yes':
        image_path = input("Enter the path of your lab test image: ").strip()
        image = cv2.imread(image_path)
        reader = easyocr.Reader(['ru'])
        results = reader.readtext(image)
        extracted_texts = [text.strip() for (_, text, _) in results]

        for i, text in enumerate(extracted_texts):
            for feature in lab_features.keys():
                if feature in text:
                    if i + 1 < len(extracted_texts):  
                        next_value = extracted_texts[i + 1]
                        match = re.search(r"\d+[\.,]?\d*", next_value)
                        if match:
                            lab_features[feature] = match.group()
    return lab_features


def predict_diagnosis_nlp():
    print("Welcome to the NLP-based diagnosis predictor!")
    user_data = collect_data_nlp()
    lab_features = extract_lab_features()
    combined_data = {**lab_features, **user_data}
    numeric_features = {feature: float(value) for feature, value in zip(lab_features.keys(), lab_features.values())}
    combined_data = {**numeric_features, **user_data}

    if not user_data:
        return

    # normalize combined_data
    input_features = np.array(list(combined_data.values())).reshape((1, 1, len(combined_data)))
    input_features = (input_features - np.min(input_features)) / (np.max(input_features) - np.min(input_features) + 1e-6)

    prediction = model.predict(input_features)
    predicted_class = np.argmax(prediction[0])
    class_name = label_encoders['Diagnosis'].classes_[predicted_class]

    print(combined_data)
    print("\nDiagnosis Prediction:")
    print(f"Predicted Class: {class_name}")
    print(f"Probabilities: {prediction[0].tolist()}")

if __name__ == "__main__":
    predict_diagnosis_nlp()

# images\FLU\2.jpeg