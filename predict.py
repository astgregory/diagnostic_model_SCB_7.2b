import joblib

model = joblib.load('models/model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

def predict_failure(input_text):
    input_vectorized = vectorizer.transform([input_text])
    prediction = model.predict(input_vectorized)
    return label_encoder.inverse_transform(prediction)[0]

if __name__ == "__main__":
    test_input = input("Введите неисправность: ")
    result = predict_failure(test_input)
    print(f"Предсказанный ответ: {result}")
