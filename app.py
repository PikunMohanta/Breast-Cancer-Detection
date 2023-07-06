from flask import Flask, render_template, request
import joblib

app = Flask(__name__,template_folder='template')

# Load the pre-trained model
model = joblib.load("breast_cancer_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the feature values from the form
    features = [float(x) for x in request.form.values()]
    # Convert the features into a 2D array
    input_features = [features]

    # Make predictions using the loaded model
    prediction = model.predict(input_features)
    
    # Map the prediction result to a human-readable label
    if prediction[0] == 0:
        result = "Benign"
    else:
        result = "Malignant"

    return render_template("index.html", prediction_result=result)

if __name__ == "__main__":
    app.run(debug=True)
