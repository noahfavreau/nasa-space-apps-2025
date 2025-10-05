import uuid
from flask import Flask, Request, jsonify, request
import pandas as pd
import inference
from preprocess import preprocess_api_input  # from src/backend/preprocess.py
from shap_generator import generate_shap_analysis  # from src/backend/shap_generator.py

app = Flask(__name__, template_folder="src/html")

model = inference.load_classifier("../model")


@app.route("/", methods=["GET"])
def main_route():
    return jsonify({"id": True}), 404


# API: /api/prediction/preditiondata
@app.route("/api/prediction/predictiondata", methods=["GET", "POST"])
def preditiondata():  # name follows route; fix typo if desired
    global model
    """
    GET: return recent prediction data
    POST: accept JSON payload and return stored/echoed prediction result
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    for i in [
        "orbital_period",
        "stellar_radius",
        "rate_of_ascension",
        "declination",
        "transit_duration",
        "transit_depth",
        "planet_radius",
        "planet_temperature",
        "insolation_flux",
        "stellar_temperature",
    ]:
        if i not in data:
            return jsonify({"error": "Missing data in JSON"}), 400

    # processing vient ici
    return jsonify(model.predict_from_raw_features(list(data.values()))), 200


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


@app.route("/api/report/shap", methods=["POST"])
def generate_shap_graph():
    global model
    """
    Generate SHAP analysis for the provided exoplanet data
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON"}), 400
        
        # Validate required fields
        required_fields = [
            "orbital_period", "stellar_radius", "rate_of_ascension", "declination",
            "transit_duration", "transit_depth", "planet_radius", "planet_temperature",
            "insolation_flux", "stellar_temperature"
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        X_inf = pd.DataFrame([data])
        
        # Generate SHAP analysis
        result = generate_shap_analysis(model, X_inf)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"SHAP analysis failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080, ssl_context="adhoc")
