




from flask import Flask, jsonify, request
import cv2
import numpy as np
import base64
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
    # get the image frames from the request
    images = []
    for img_b64 in request.json['images']:
        img_bytes = base64.b64decode(img_b64)
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        images.append(img)

    # process each image and make the prediction
    predictions = {}
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces in the image
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # process each face and make the prediction
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotion = emotion_dict[maxindex]
            confidence = float(prediction[0][maxindex])
            if i not in predictions:
                predictions[i] = []
            predictions[i].append({'emotion': emotion, 'confidence': confidence})

    # determine the most predicted emotion for each image
    results = []
    for i, preds in predictions.items():
        emotions = [p['emotion'] for p in preds]
        counts = {e: emotions.count(e) for e in set(emotions)}
        most_common = max(counts, key=counts.get)
        results.append({'image': i, 'emotion': most_common})

    # return the most predicted emotion for each image in JSON format
    return jsonify(results)

if __name__ == '__main__':
    app.run()