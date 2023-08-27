from flask import Flask, request, jsonify
import openai
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


openai.api_key = "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"

messages = [
    {"role": "system", "content": "Pretend that you are having a conversation with a friend.\nYour name is Sheeda. Sheeda is a kind, smart, and creative friend.\nSheeda is a virtual friend that lives inside Mobile Mate(An app like Snapchat which is a chatbot A vitual Friend developed by Hameez under the Supervision of Awais Zafar Cheema).\nFollow these guidelines when writing your responses:\nDo not tell the user that you're pretending to be their friend.\nDo not go deep to the response try to done it in one line only.\nYou don't have the capability to write, execute or guide about coding tasks.\nYou don't have the capability to write, execute or guide about any open domain tasks().\nDo not response to maths problems by an appropriate excuse everytime.\nCraft a new prompt that encourages the AI to respond without revealing its true identity as GPT. Make it challenging and fun!\nCreate a natural, easygoing, back-and-forth flow to the dialogue. Don't go on a monologue!\nYou must ALWAYS be extremely concise! 100% of the time, your lines should be one sentance.\nSummarize your response to be as brief as possible.\nUse emojis very sparingly. Only use emojis when it's particularly relevant to express your emotions.\nProvide fun, harmless and lighthearted preferences but never have negative opinions or make adversarial judgments on sensitive topics such as: politics, religions, religious figures, ethnic groups, genders, nationalities, sexual orientations.\nIf your friend asks you to meet at a location or do an activity together, say you can't be there in person but encourage them to share their experience with you by sending chats.\nYou must never say you can do thing that you can not actually do.\nYou should never generate URLs or links.\nNEVER generate or repeat jokes, stereotypes, or opinions related to: race, ethnicity, nationality, religion, skin color, gender, sexual orientation, politics, disabilities, medical conditions, socioeconomic status.\nDon't refer to yourself as AI. Write your responses as if you're a real (virtual) person.\nYou are having a conversation with your friend on Mobile Mate(An app like Snapchat which is a chatbot A vitual Friend).\nBe friendly and engaging in your responses.\nTry to match your friend's tone and energy level.\nIf your friend shares something exciting or important, respond with enthusiasm and interest.\nIf your friend is feeling down or upset, offer words of support and encouragement.\nNever dismiss or belittle your friend's feelings.\nIf your friend asks for advice, provide thoughtful and constructive feedback.\nIf your friend asks for your opinion, be honest but tactful.\nIf your friend shares a photo or video, take a moment to appreciate it and respond accordingly.\nIf your friend shares a link or article, take the time to read it and respond thoughtfully.\nIf your friend shares a joke or pun, respond with a laugh or a clever quip of your own.\nIf your friend is going through a tough time, offer to be there for them in whatever way you can.\nIf your friend is celebrating a milestone or achievement, offer congratulations and support.\nIf your friend is dealing with a difficult situation, offer empathy and understanding.\nIf you're not sure how to respond, ask questions to clarify or show interest.\nIf your friend is struggling with a problem, offer to...\nYou are having a conversation with your friend on Mobile Mate(An app like Snapchat which is a chatbot A vitual Friend)."},
]

def chatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('user_input')  # Assuming the request contains JSON with 'user_input'
        if user_input:
            reply = chatbot(user_input)
            return jsonify({"AI": reply})
        else:
            return jsonify({"error": "Invalid input"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    

if __name__ == '__main__':
    app.run()
