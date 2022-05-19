# Word Predictor

Basic Dash interface for word prediction.

## Installation

```
pip install -r requirements.txt
```

## Models

The predictor is based on two different NLP technologies:
- an n-gram model
- a Recurrent Neural Network (RNN)

## Run the interface

To interact with the algorithm, we developped a Dash web application. To run it, simply run the `app.py` file as follow:

```
python app.py
```

Then, go on your browser and type `127.0.0.1:8050` to access the app. You can then start writing in the text area and words are going to be suggested to you in the console. If you click on one of them, it will automatically add the word at the end of your text.