"""
train_model.py — Trains a Sentiment Analysis model using Scikit-Learn
and saves it as model.pkl for use by the Flask backend.
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

positives = [
    "I absolutely loved this movie, it was fantastic!",
    "The food was delicious and the service was excellent.",
    "What a wonderful experience, I had a great time.",
    "This product exceeded all my expectations.",
    "Amazing performance by the entire cast.",
    "The book was beautifully written and deeply moving.",
    "I'm so happy with this purchase, highly recommend!",
    "Outstanding quality and fast delivery.",
    "The staff were incredibly kind and helpful.",
    "Best experience I've had in years, truly spectacular.",
    "Brilliant work, very impressed with the results.",
    "This is the best app I have ever used.",
    "Phenomenal taste, I will definitely come back.",
    "Loved every moment, couldn't put it down.",
    "Perfect gift, my friend was absolutely thrilled.",
    "Smooth, fast, and incredibly intuitive to use.",
    "The concert was breathtaking and unforgettable.",
    "Superb craftsmanship and attention to detail.",
    "I feel so refreshed and energized after using this.",
    "A masterpiece of storytelling, loved every chapter.",
    "Great customer support, resolved my issue instantly.",
    "Highly polished and professional, exceeded expectations.",
    "Wonderful atmosphere, great value for money.",
    "The instructor was patient, clear, and very knowledgeable.",
    "Exceeded every expectation, simply outstanding!",
    "Really enjoyed this, would definitely do it again.",
    "Such a pleasant surprise, better than I expected.",
    "The quality is top notch, very satisfied.",
    "Incredibly well made, feels premium and sturdy.",
    "Delightful experience from start to finish.",
    "Very happy with my purchase, works perfectly.",
    "Fantastic results, I am genuinely impressed.",
    "Love the design, it looks and feels amazing.",
    "The team was so professional and friendly.",
    "Would absolutely recommend this to all my friends.",
    "Such good value, way better than competitors.",
    "Arrived quickly and in perfect condition.",
    "Everything worked exactly as described, very pleased.",
    "I was blown away by how good this is.",
    "Five stars, no complaints whatsoever.",
    "This made my day so much better, thank you!",
    "Really high quality stuff, worth every penny.",
    "So glad I bought this, life changing honestly.",
    "The best I have ever tried, no question.",
    "Super easy to use and the results are great.",
    "Genuinely one of the best purchases I have made.",
    "Awesome product, the whole family loves it.",
    "This exceeded my hopes, I am thrilled.",
    "Impressive attention to detail, clearly made with care.",
    "The experience was seamless and enjoyable throughout.",
    "I smiled the entire time, absolutely loved it.",
    "Nothing but good things to say about this.",
    "Works like a charm, exactly what I needed.",
    "Beautiful, elegant, and incredibly well thought out.",
    "Customer service went above and beyond for me.",
    "So impressed with how quickly everything was resolved.",
    "This is exactly what I was looking for.",
    "I feel great using this every single day.",
    "The packaging alone was impressive, great first impression.",
    "Solid, reliable, and does exactly what it promises.",
    "My whole family is obsessed with this now.",
    "Can't imagine life without it at this point.",
    "Refreshing to find something that actually works well.",
    "Very thoughtful design, clearly built for the user.",
    "Loved the ambiance, the staff, and the food.",
    "Worth every single cent, highly satisfied customer.",
    "Absolutely perfect for what I needed it for.",
    "Got so many compliments after using this product.",
    "An absolute joy to use from the first day.",
    "This brand never disappoints, consistently excellent.",
    "Super impressed with the build quality and finish.",
    "Would give six stars if I could honestly.",
    "Made exactly as advertised, no surprises, just great.",
    "Prompt delivery, great quality, happy customer here.",
    "This genuinely put a smile on my face.",
    "I feel so confident recommending this to everyone.",
]

negatives = [
    "This was the worst movie I have ever seen.",
    "Terrible service and the food was cold and bland.",
    "I am extremely disappointed with this product.",
    "The quality is awful and it broke after one day.",
    "A complete waste of money, do not buy this.",
    "The staff were rude and completely unhelpful.",
    "Horrible experience, I will never return here.",
    "Very slow delivery and the packaging was damaged.",
    "Absolutely dreadful, nothing worked as advertised.",
    "Poor quality and the instructions were confusing.",
    "I regret purchasing this, total waste of time.",
    "Nothing worked as described, very frustrating.",
    "The worst customer service I have ever encountered.",
    "Terrible, fell apart within a week of use.",
    "Misleading description, the product looks nothing like the photo.",
    "Awful smell, gave me a headache immediately.",
    "Poorly made, very cheap and flimsy materials.",
    "Extremely overpriced for such low quality.",
    "The app crashes constantly and loses all my data.",
    "Rude staff and a very unpleasant environment.",
    "Boring and completely predictable, not worth watching.",
    "Shocking quality, returned it immediately.",
    "Useless instructions and the product does not work at all.",
    "Felt cheated, this is nothing like what was advertised.",
    "Worst purchase decision I have ever made.",
    "I would not recommend this to my worst enemy.",
    "Genuinely awful, I want my money back.",
    "Stopped working after two uses, complete garbage.",
    "The smell was unbearable, had to throw it away.",
    "I have never been so disappointed in a product.",
    "Overpriced, underdelivered, and a total letdown.",
    "The instructions made no sense and nothing worked.",
    "Cheap plastic that broke the moment I opened it.",
    "This is a scam, the product is completely fake.",
    "I waited three weeks and it still arrived broken.",
    "Absolutely disgusting experience, never again.",
    "The movie was painfully boring and way too long.",
    "Worst meal I have ever had, completely inedible.",
    "Service was shockingly bad, waited over an hour.",
    "Zero stars if I could, truly terrible product.",
    "Looks nothing like the pictures, very misleading.",
    "I am furious, this is not what I ordered.",
    "The staff ignored me and were completely dismissive.",
    "This is the last time I buy from this brand.",
    "Completely useless, does not do what it claims.",
    "I threw it in the bin after one day.",
    "Very poor build quality, feels like a toy.",
    "The website was buggy and my order got lost.",
    "Ripped off, this product is a total fraud.",
    "Never buying from here again, absolutely dreadful.",
    "Everything about this experience was unpleasant.",
    "The taste was revolting, I could not finish it.",
    "Such a letdown, I had high hopes but it failed.",
    "Unreliable, inconsistent, and not worth the price.",
    "It arrived late, damaged, and missing parts.",
    "I asked for help and was rudely turned away.",
    "This product made my problem worse, not better.",
    "Buyer beware, this is not worth your money.",
    "I have had better experiences at far cheaper places.",
    "The battery died after an hour, absolutely useless.",
    "Do not waste your time or money on this.",
    "Really unpleasant from start to finish.",
    "It looked good online but in reality it is trash.",
    "Stopped working after the first week, so frustrating.",
    "The refund process was a nightmare on top of it all.",
    "I left feeling worse than when I arrived.",
    "The quality has gone downhill so much recently.",
    "I got sick from the food, never going back.",
    "The whole thing felt rushed and poorly thought out.",
    "Not fit for purpose, completely failed to deliver.",
    "One of the worst decisions I have made this year.",
    "Falling apart already, and I only got it yesterday.",
    "The worst value for money I have ever experienced.",
    "I cannot believe they charge this much for this.",
    "Avoid at all costs, you will regret it otherwise.",
]

texts = positives + negatives
labels = (["positive"] * len(positives)) + (["negative"] * len(negatives))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        stop_words='english',
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        C=2.0,
        solver='lbfgs',
        class_weight='balanced'
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

pipeline.fit(X_train, y_train)

print("=== Model Evaluation ===")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved as model.pkl")
