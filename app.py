"""
app.py — Flask backend using VADER for sentiment analysis.
VADER understands intensity words like "very", "extremely", "a little"
which TF-IDF based models cannot differentiate.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
CORS(app)

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field.'}), 400

        text = clean_text(data['text'])

        if len(text) < 3:
            return jsonify({'error': 'Text is too short.'}), 400

        if len(text) > 2000:
            return jsonify({'error': 'Text is too long.'}), 400

        scores = analyzer.polarity_scores(text)

        # compound score ranges from -1 (most negative) to +1 (most positive)
        compound = scores['compound']

        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        # convert to percentages for the frontend
        pos_score = round(scores['pos'] * 100, 1)
        neg_score = round(scores['neg'] * 100, 1)
        neu_score = round(scores['neu'] * 100, 1)

        abs_compound = abs(compound)
        if compound >= 0.05:
            # positive: mild -> 60-70%, extreme -> 80-90%
            confidence = 60 + ((abs_compound - 0.05) / (1.0 - 0.05)) * 30
            confidence = round(min(confidence, 90), 1)
        elif compound <= -0.05:
            # negative: mild -> 30-40%, extreme -> 20-30%
            # more negative = lower confidence, so we invert
            confidence = 40 - ((abs_compound - 0.05) / (1.0 - 0.05)) * 20
            confidence = round(max(confidence, 20), 1)
        else:
            confidence = 50.0  # neutral zone

        return jsonify({
            'label':      label,
            'confidence': confidence,
            'pos_score':  pos_score,
            'neg_score':  neg_score,
            'neu_score':  neu_score,
            'compound':   round(compound, 4),
            'text':       text,
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
