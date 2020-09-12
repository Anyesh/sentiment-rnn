import time
import flask
from flask import jsonify, render_template, request
from enigne import predict_sentiment

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()
        payload = request.get_json()
        prediction = predict_sentiment(payload)
        end_time = time.time() - start_time
        # prediction = 0.9
    except Exception as e:
        return jsonify({"invalid": "invalid request.."})
    return jsonify(
        {
            "prediction_score": prediction
            if float(prediction) > 0.5
            else 1 - float(prediction),
            "status": "NEGATIVE" if float(prediction) > 0.5 else "POSITIVE",
            "time_taken": str(end_time),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
