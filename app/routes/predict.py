from flask import Blueprint, request, jsonify, send_file, current_app
import os
from ..services.prediction_service import PredictionService

predict_bp = Blueprint("predict", __name__)
_service = PredictionService() 

@predict_bp.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "image is required"}), 400

    result = _service.predict_with_explain(file, current_app.config)
    return jsonify({
        "label": result["label"],
        "confidence": result["confidence"],
        "image_url": f"/overlay/{result['overlay_id']}"
    })

@predict_bp.route("/overlay/<overlay_id>", methods=["GET"])
def overlay(overlay_id):
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{overlay_id}.jpg")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/jpeg")
