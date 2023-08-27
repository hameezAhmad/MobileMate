from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def welcome():
    return jsonify({ "message": "Welcome to my API!" })

if __name__ == '__main__':
    app.run(host='192.168.8.205', port=5000)