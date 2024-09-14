from flask import Flask, request, jsonify
import subprocess
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To allow communication with React frontend

# Example route for prediction
@app.route('/predict/<model_name>', methods=['POST'])
def run_prediction(model_name):
    try:
        # Example: Load input data from request
        input_data = request.get_json()

        # Run the appropriate notebook based on model_name
        if model_name == "future":
            result = run_notebook('models/future_prediction.ipynb', input_data)
        elif model_name == "barley":
            result = run_notebook('models/barley_crop_prediction.ipynb', input_data)
        elif model_name == "rice":
            result = run_notebook('models/rice_crop_files_prediction.ipynb', input_data)
        else:
            return jsonify({"error": "Model not found"}), 404

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to run the Jupyter notebook
def run_notebook(notebook_path, input_data):
    # You can pass the input data to your notebook here as needed
    command = f"jupyter nbconvert --to notebook --execute {notebook_path} --output output_notebook.ipynb"
    subprocess.run(command, shell=True, check=True)

    # Example: Load some prediction result (replace this with your actual output)
    result = pd.read_csv('data/soya_data.csv')  # Assuming CSV as output
    return result.head().to_dict()

if __name__ == '__main__':
    app.run(debug=True)
