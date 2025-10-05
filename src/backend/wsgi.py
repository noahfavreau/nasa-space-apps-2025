import uuid
from flask import Flask, Request, jsonify, request
import pandas as pd

from preprocess import preprocess_api_input  # from src/backend/preprocess.py
from shap_generator import generate_shap_analysis  # from src/backend/shap_generator.py

app = Flask(__name__, template_folder="src/html")


@app.route("/", methods=["GET"])
def main_route():
    return jsonify({"id": "1"})


# API: /api/prediction/preditiondata
@app.route("/api/prediction/preditiondata", methods=["GET", "POST"])
def preditiondata(data: Request):  # name follows route; fix typo if desired
    """
    GET: return recent prediction data
    POST: accept JSON payload and return stored/echoed prediction result
    """

    request.files.get("csv_file")
    # processing vient ici

    return jsonify({"examples": {}}), 200


# API: /api/prediction/graph
@app.route("/api/prediction/graph", methods=["GET"])
def prediction_graph():
    return jsonify("{}"), 200


# API: /api/prediction/fillexample
@app.route("/api/prediction/fillexample/<example_id>", methods=["GET"])
def fillexample(example_id):
    """
    Return a sample payload clients can use to prefill forms.
    """
    """TODO
    fill ça avec les vraies données
    """
    return_value = {"id": example_id}
    return jsonify(return_value), 200


@app.route("/api/report/shap", methods=["GET"])
def generate_shap_graph():
    return jsonify(generate_shap_analysis())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080, ssl_context="adhoc")
