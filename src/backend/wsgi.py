import uuid
from flask import Flask, Request, jsonify, request
from flask_cors import CORS
import pandas as pd
import inference
from preprocess import preprocess_api_input  # from src/backend/preprocess.py
from shap_generator import generate_shap_analysis  # from src/backend/shap_generator.py
from pathlib import Path
from preprocess import preprocess_api_input
from shap_generator import generate_shap_analysis
from shap import Explainer


app = Flask(__name__, template_folder="src/html")
CORS(
    app,
    origins=[
        "https://www.bottomlessswag.tech",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

model = inference.load_classifier("../model")


@app.route("/api/restart", methods=["GET"])
def restart():
    p = Path("/tmp/reboot.txt")
    p.touch(exist_ok=True)
    return "Will reboot soon"


@app.route("/", methods=["GET"])
def main_route():
    return jsonify({"id": True}), 404


# API: /api/prediction/preditiondata
@app.route("/api/prediction/predictiondata", methods=["GET", "POST"])
def preditiondata():
    global model
    """
    GET: return recent prediction data
    POST: accept JSON payload and return stored/echoed prediction result
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    # Map "insolation_flux" to "insolation flux" for API compatibility FIRST
    if "insolation_flux" in data:
        data["insolation flux"] = data.pop("insolation_flux")

    # Now validate that all required fields are present
    required_fields = [
        "orbital_period",
        "stellar_radius",
        "rate_of_ascension",
        "declination",
        "transit_duration",
        "transit_depth",
        "planet_radius",
        "planet_temperature",
        "insolation flux",
        "stellar_temperature",
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing data in JSON", 
            "missing_fields": missing_fields,
            "hint": "Use 'insolation_flux' (underscore) or 'insolation flux' (space)"
        }), 400

    ordered_values = [data[field] for field in required_fields]
    return jsonify(model.predict_from_raw_features(ordered_values)), 200


# API: /api/prediction/bulk
@app.route("/api/prediction/bulk", methods=["POST"])
def bulk_prediction():
    """
    Handle bulk predictions from JSON data or file uploads
    """
    global model
    try:
        # Check if it's a file upload or JSON data
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            
            # Process the uploaded file
            if file.filename.lower().endswith('.csv'):
                # Read CSV file
                df = pd.read_csv(file)
                has_raw_features = request.form.get('has_raw_features', 'true').lower() == 'true'
                
                # Process the data
                result = model.predict_batch_from_dataframe(df, has_raw_features=has_raw_features)
                
                if result['success']:
                    return jsonify({
                        "success": True,
                        "data": {
                            "predictions": result['data'].to_dict('records'),
                            "statistics": result['statistics']
                        },
                        "error": None
                    }), 200
                else:
                    return jsonify(result), 500
                    
            elif file.filename.lower().endswith('.json'):
                # Read JSON file
                import json
                content = file.read().decode('utf-8')
                data = json.loads(content)
                
                # Handle single object or array of objects
                if isinstance(data, dict):
                    data = [data]
                
                predictions = []
                for i, item in enumerate(data):
                    try:
                        # Map insolation_flux to insolation flux if needed
                        if "insolation_flux" in item:
                            item["insolation flux"] = item.pop("insolation_flux")
                        
                        # Extract ordered values with validation
                        required_fields = [
                            "orbital_period", "stellar_radius", "rate_of_ascension", "declination",
                            "transit_duration", "transit_depth", "planet_radius", "planet_temperature",
                            "insolation flux", "stellar_temperature"
                        ]
                        
                        # Check for missing fields
                        missing_fields = [field for field in required_fields if field not in item]
                        if missing_fields:
                            return jsonify({
                                "success": False,
                                "error": f"Missing fields in item {i+1}: {missing_fields}",
                                "hint": "Use 'insolation_flux' (underscore) or 'insolation flux' (space)"
                            }), 400
                        
                        ordered_values = [item[field] for field in required_fields]
                        pred_result = model.predict_from_raw_features(ordered_values)
                        predictions.append(pred_result)
                        
                    except Exception as e:
                        return jsonify({
                            "success": False,
                            "error": f"Error processing item {i+1}: {str(e)}"
                        }), 500
                
                return jsonify({
                    "success": True,
                    "data": {
                        "predictions": predictions,
                        "total_processed": len(predictions)
                    },
                    "error": None
                }), 200
            else:
                return jsonify({"success": False, "error": "Unsupported file format"}), 400
        
        else:
            # Handle JSON data in request body
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400
            
            # Handle single object or array of objects
            if isinstance(data, dict):
                data = [data]
            
            predictions = []
            for item in data:
                # Map insolation_flux to insolation flux if needed
                if "insolation_flux" in item:
                    item["insolation flux"] = item.pop("insolation_flux")
                
                # Extract ordered values
                required_fields = [
                    "orbital_period", "stellar_radius", "rate_of_ascension", "declination",
                    "transit_duration", "transit_depth", "planet_radius", "planet_temperature",
                    "insolation flux", "stellar_temperature"
                ]
                ordered_values = [item[field] for field in required_fields]
                pred_result = model.predict_from_raw_features(ordered_values)
                predictions.append(pred_result)
            
            return jsonify({
                "success": True,
                "data": {
                    "predictions": predictions,
                    "total_processed": len(predictions)
                },
                "error": None
            }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "data": None,
            "error": str(e)
        }), 500


# API: /api/prediction/bulk/csv
@app.route("/api/prediction/bulk/csv", methods=["POST"])
def bulk_prediction_csv():
    """
    Bulk prediction with CSV output for download
    """
    global model
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"success": False, "error": "Only CSV files are supported for this endpoint"}), 400
        
        # Read the uploaded CSV
        df = pd.read_csv(file)
        has_raw_features = request.form.get('has_raw_features', 'true').lower() == 'true'
        
        # Process with the classifier
        result = model.predict_batch_from_dataframe(df, has_raw_features=has_raw_features)
        
        if result['success']:
            # Convert results to CSV format for response
            from flask import make_response
            import io
            
            output = io.StringIO()
            result['data'].to_csv(output, index=False)
            output.seek(0)
            
            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = f"attachment; filename=predictions_{file.filename}"
            response.headers["Content-type"] = "text/csv"
            
            return response
        else:
            return jsonify(result), 500
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# API: /api/prediction/status
@app.route("/api/prediction/status", methods=["GET"])
def prediction_status():
    """
    Get API status and model information
    """
    global model
    try:
        model_info = model.get_model_info()
        return jsonify({
            "success": True,
            "status": "ready",
            "model_info": model_info
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "error",
            "error": str(e)
        }), 500


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
            "orbital_period",
            "stellar_radius",
            "rate_of_ascension",
            "declination",
            "transit_duration",
            "transit_depth",
            "planet_radius",
            "planet_temperature",
            "insolation flux",
            "stellar_temperature",
        ]

        # Map "insolation_flux" to "insolation flux" if necessary for API compatibility
        if "insolation_flux" in data:
            data["insolation flux"] = data.pop("insolation_flux")

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        ordered_values = [data[field] for field in required_fields]
        X_inf = pd.DataFrame([ordered_values], columns=required_fields)

        # --- FIX: Use a base model for SHAP, not the meta-model ---
        # For SHAP analysis, we'll use a base model since meta-model requires transformed features
        # Use first available base model (prioritize XGBoost for better SHAP support)
        shap_model = None
        
        # Try XGBoost first
        if hasattr(model, "base_models") and "xgboost" in model.base_models:
            base_models = model.base_models["xgboost"]
            if base_models and len(base_models) > 0:
                shap_model = base_models[0]
                print("Using XGBoost base model for SHAP analysis")
        
        # Try CatBoost as fallback
        elif hasattr(model, "base_models") and "catboost" in model.base_models:
            base_models = model.base_models["catboost"]
            if base_models and len(base_models) > 0:
                shap_model = base_models[0]
                print("Using CatBoost base model for SHAP analysis")
        
        # Try LightGBM as another fallback
        elif hasattr(model, "base_models") and "lightgbm" in model.base_models:
            base_models = model.base_models["lightgbm"]
            if base_models and len(base_models) > 0:
                shap_model = base_models[0]
                print("Using LightGBM base model for SHAP analysis")

        if shap_model is None:
            return jsonify({"error": "No suitable base model available for SHAP analysis"}), 500

        # Run SHAP analysis on selected model
        from shap_generator import generate_shap_analysis

        result = generate_shap_analysis(shap_model, X_inf)

        return jsonify(result), 200

    except Exception as e:
        return jsonify(
            {"success": False, "error": f"SHAP analysis failed: {str(e)}"}
        ), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
