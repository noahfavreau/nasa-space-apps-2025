import uuid
from flask import Flask, jsonify, request, render_template, render_template_string
import pandas as pd
import shap
import os

app = Flask(__name__, template_folder="src/html")


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
    return jsonify({"examples": {}}), 200


# API: /api/prediction/graph
@app.route("/api/prediction/graph", methods=["GET"])
def prediction_graph():
    """
    Return JSON suitable for client-side graphing (labels + values).
    """

    return jsonify("{}"), 200


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
    return render_template_string(get_file_as_string("equipe.html"))


# Page: /workspace
@app.route("/workspace", methods=["GET"])
def workspace():
    return render_template_string(get_file_as_string("prediction.html"))


@app.route("/")
def landing():
    return render_template_string(get_file_as_string("index.html"))


def get_file_as_string(path: str):
    a = os.path.join("src/html", path)
    with open(a, "r") as f:
        return f.read()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080, ssl_context="adhoc")
