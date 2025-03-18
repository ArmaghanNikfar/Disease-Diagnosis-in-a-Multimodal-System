from textblob import TextBlob

text = "yed i hare not"
blob = TextBlob(text)
corrected_text = str(blob.correct())
print(corrected_text)  # خروجی: "no i have not"