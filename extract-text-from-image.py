import easyocr
import cv2
import re

def extract_lab_features():
    image_path = input("Enter the path of your lab test image: ").strip()
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['ru'])
    results = reader.readtext(image)
    extracted_texts = [text.strip() for (_, text, _) in results]

    feature_list = [
        'Эпителий плоский', 'Лейкоциты', 'С-реактивный белок (СРБ)',
        'Общий Белок', 'Тромбоциты', 'Лимфоциты'
    ]

    feature_values = {}

    for i, text in enumerate(extracted_texts):
        for feature in feature_list:
            if feature in text:
                if i + 1 < len(extracted_texts):  
                    next_value = extracted_texts[i + 1]
                    match = re.search(r"\d+[\.,]?\d*", next_value)
                    if match:
                        feature_values[feature] = match.group()

    return feature_values

# normal:  Эпителий плоский : 0-4  Лейкоциты : 3.8-8.6  С-реактивный белок (СРБ): 0.08-5 Общий Белок: 66-87 Тромбоциты : 180-320  Лимфоциты : 19-39