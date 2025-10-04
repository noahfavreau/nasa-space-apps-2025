import uuid
from flask import Flask, jsonify, request, render_template_string
import pandas as pd

app = Flask(__name__)


# Apply the default theme


# Example in-memory example data (replace with DB or real logic)
EXAMPLE_PREDICTION = {
    "id": 1,
    "input": {"feature_a": 10, "feature_b": 5},
    "prediction": 0.78,
    "timestamp": "2025-10-04T00:00:00Z",
}


# API: /api/prediction/preditiondata
@app.route("/api/prediction/preditiondata", methods=["GET", "POST"])
def preditiondata():  # name follows route; fix typo if desired
    """
    GET: return recent prediction data
    POST: accept JSON payload and return stored/echoed prediction result
    """
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        # Placeholder: normally run model/persistence here
        result = {
            "received": payload,
            "prediction": 0.5,  # replace with model output
            "status": "ok",
        }
        return jsonify(result), 201
    # GET
    return jsonify({"examples": [EXAMPLE_PREDICTION]}), 200


# API: /api/prediction/graph
@app.route("/api/prediction/graph", methods=["GET"])
def prediction_graph():
    """
    Return JSON suitable for client-side graphing (labels + values).
    """

    return jsonify(GRAPH_SAMPLE), 200


# API: /api/prediction/fillexample
@app.route("/api/prediction/fillexample", methods=["GET"])
def fillexample():
    """
    Return a sample payload clients can use to prefill forms.
    """
    sample = None
    """TODO
    fill ça avec les vraies données
    """
    return jsonify(sample), 200


# Page: /about
@app.route("/about", methods=["GET"])
def about():
    html = """
    <!doctype html>
    <html>
      <head><title>About</title></head>
      <body>
        <h1>About</h1>
        <p>Simple Flask app exposing prediction APIs and workspace page.</p>
      </body>
    </html>
    """
    return render_template_string(html)


# Page: /workspace
@app.route("/workspace", methods=["GET"])
def workspace():
    # Minimal interactive page demonstrating fetch to API endpoints
    html = None
    return render_template_string(html)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
