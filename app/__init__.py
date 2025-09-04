from flask import Flask
from flask_cors import CORS
from .config import Config
from .routes.predict import predict_bp
import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    CORS(app)

    upload = app.config.get("UPLOAD_FOLDER", "uploads")
    if not os.path.isabs(upload):
        upload = os.path.abspath(os.path.join(app.root_path, "..", upload))
    app.config["UPLOAD_FOLDER"] = upload
    os.makedirs(upload, exist_ok=True)

    app.register_blueprint(predict_bp, url_prefix="/")
    return app
