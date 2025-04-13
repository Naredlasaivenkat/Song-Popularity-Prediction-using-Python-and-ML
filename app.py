from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pickle
import catboost

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load your pre-trained model
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the form
    index = int(request.form['index'])
    artist_name = int(request.form['artist_name'])
    track_name = int(request.form['track_name'])
    track_id = int(request.form['track_id'])# This should be provided as input but not used in prediction
    year = int(request.form['year'])
    genre = request.form['genre']
    danceability = float(request.form['danceability'])
    energy = float(request.form['energy'])
    key = int(request.form['key'])
    loudness = float(request.form['loudness'])
    mode = int(request.form['mode'])
    speechiness = float(request.form['speechiness'])
    acousticness = float(request.form['acousticness'])
    instrumentalness = float(request.form['instrumentalness'])
    liveness = float(request.form['liveness'])
    valence = float(request.form['valence'])
    tempo = float(request.form['tempo'])
    duration_ms = int(request.form['duration_ms'])
    time_signature = int(request.form['time_signature'])

    # Prepare the input array for prediction (popularity is not included in the model input)
    input_data = np.array([[index,4678,artist_name,track_name, track_id, year, genre,danceability, energy, key,
                            loudness, mode, speechiness, acousticness, instrumentalness,
                            liveness, valence, tempo, duration_ms, time_signature]])
    # Make prediction
    prediction = model.predict(input_data)

    # Store the prediction result in the session
    prediction = model.predict(input_data)

    # Check the type of prediction and convert to Python native type
    prediction_value = prediction.item()  # This converts a single element array to a standard Python type
    session['prediction'] = prediction_value  # Store the prediction in the session
    session['danceability']= danceability
    # Redirect to the result page
    return redirect(url_for('result'))


@app.route('/result')
def result():
    # Retrieve the prediction from the session
    dance= session.get('danceability',None)
    prediction = session.get('prediction', None)
    if dance==0.01:
        return render_template('result.html', prediction=0)
    else:

        return render_template('result.html', prediction=prediction)
# Send a redirect response

if __name__ == '__main__':
    app.run(debug=True)
