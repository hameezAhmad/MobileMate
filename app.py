
from flask import Flask, jsonify, request
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# load the pre-trained model
model = keras.models.load_model("my_model.h5")

# load the haar cascade
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    # get the image from the request
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # process each face and make the prediction
    predictions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]
        confidence = float(prediction[0][maxindex])
        predictions.append({'emotion': emotion, 'confidence': confidence})

    # return the predictions in JSON format
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='192.168.14.205', port=5000)

# from flask import Flask, jsonify, request
# import cv2
# import numpy as np
# import tensorflow as tf

# app = Flask(__name__)

# # load the TFLite model
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # load the haar cascade
# facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # define the emotion dictionary
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# @app.route('/predict_emotion', methods=['POST'])
# def predict_emotion():
#     # get the image from the request
#     file = request.files['image'].read()
#     npimg = np.frombuffer(file, np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # detect faces in the image
#     faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # process each face and make the prediction
#     predictions = []
#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]
#         cropped_img = cv2.resize(roi_gray, (48, 48))
#         input_data = np.expand_dims(np.expand_dims(cropped_img, -1), 0).astype(np.float32)
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         prediction = interpreter.get_tensor(output_details[0]['index'])
#         maxindex = int(np.argmax(prediction))
#         emotion = emotion_dict[maxindex]
#         confidence = float(prediction[0][maxindex])
#         predictions.append({'emotion': emotion, 'confidence': confidence})

#     # return the predictions in JSON format
#     return jsonify(predictions)

# if __name__ == '__main__':
#     app.run(host='192.168.14.205', port=5000)



