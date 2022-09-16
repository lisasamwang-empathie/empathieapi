# %%
# !pip -q install flask
# !pip -q install flask_restful
# !pip -q install tensorflow
# !pip -q install flask-ngrok
# !pip -q install pyngrok==4.1.1
# !ngrok authtoken 2EdmM1qZZJUGa5dDtZrPqQPjh9I_7QHSq7zGBcDLhkzHqDynv

# %%
# import tensorflow as tf
# print(tf.__version__)

# %%
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import numpy as np
# import pickle
import json
import tensorflow as tf
# from flask_ngrok import run_with_ngrok

# %%
# usemodel = tf.keras.models.load_model('model_flask')
# usemodel.summary()

# %%

app = Flask(__name__)
api = Api(app)
# run_with_ngrok(app)   

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define how the api will respond to the post requests
class IrisClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        X = np.array(json.loads(args["data"]))
        model = tf.keras.models.load_model('model_flask')
        # with open('model.pickle', 'rb') as f:
        #     model = pickle.load(f)
        prediction = model.predict(X)
        print(type(prediction))
        return jsonify(prediction.tolist())

api.add_resource(IrisClassifier, "/iris")

if __name__ == "__main__":
    # Load model
#     with open('model_flask', 'rb') as f:
    # model = usemodel
    # with open('model.pickle', 'rb') as f:
    #     model = pickle.load(f)
    # model = tf.keras.models.load_model('model_flask')
    app.run(port=8080)

# %%
# %%



