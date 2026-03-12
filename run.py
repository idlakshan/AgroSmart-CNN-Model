import os
print("Current working directory:", os.getcwd())
print("Files in project root:", os.listdir("."))
print("Files in model folder:", os.listdir("model"))
print("Weights path exists?", os.path.exists("model/soil_model_weights_only.weights.h5"))

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    app.run(debug=True)
