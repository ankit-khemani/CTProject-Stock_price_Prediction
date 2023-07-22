from flask import Flask, request, jsonify,render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')  

@app.route('/')
def index():
    # You can pass data from Flask to HTML using variables
    title = "Welcome to My Flask App"
    message = "Welcome to Reliance Stock Price Prediction API"
    return render_template('index.html', title=title, message=message)



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request
        data = request.get_json()
        # Assuming the input features are provided as a dictionary
        features = [data['Open'], data['High'], data['Low'], data['Volume']] 

        # Perform prediction using the loaded model
        prediction = model.predict([features])[0]

        # Create a response dictionary with the prediction result
        response = {'prediction': prediction}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
