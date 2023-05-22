import re
import string
import pickle

from inference import translate_text, get_translation_model, get_lr_model, recognize_learning_style

from flask import Flask, jsonify, request, Response, render_template


custom_punct = string.punctuation.replace("-","").replace("'","")
def clean(text):
    text = text.lower()
    text = re.sub("["+custom_punct+"]", "", text)
    return text


en_tokenizer = pickle.load(open("./tokenizers/en_tokenizer.pickle","rb"))
fr_tokenizer = pickle.load(open("./tokenizers/fr_tokenizer.pickle", "rb"))
input_vocab_size = len(en_tokenizer.index_word) + 1
output_vocab_size = len(fr_tokenizer.index_word) + 1

learning_style_tokenizer = pickle.load(open("./tokenizers/tokenizer.pickle", "rb"))


model = get_translation_model(input_vocab_size, output_vocab_size)

lr_model = get_lr_model(learning_style_tokenizer)


app = Flask(__name__, static_url_path='/static')
@app.route("/translate", methods=["POST"])
def translate():
    
    text = request.json['text']
    text = clean(text)
    text = [text]
    label = translate_text(text, model, en_tokenizer, fr_tokenizer)

    return jsonify({"result": label})


@app.route("/recognize_lr", methods=["POST"])
def recognize_lr():
    
    text = request.json['text']
    text = clean(text)
    text = [text]
    label = recognize_learning_style(text, lr_model, learning_style_tokenizer)

    return jsonify({"result": label})

@app.route("/", methods=["GET"])
def home():
    
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)


