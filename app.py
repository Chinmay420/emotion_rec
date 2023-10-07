from flask import Flask, render_template, Response, send_file, make_response
from keras.models import load_model
from time import sleep, strftime
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import csv
import io
from flask import Flask, request
import librosa
import subprocess
import concurrent.futures
import speech_recognition as sr
import time
import logging
import os
import threading
import sys
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(levelname)s - %(message)s"
)


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')
audio_classifier = load_model('Audio_.h5')  # load the audio model


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_map = {'Angry': -1, 'Disgust': -1, 'Fear': -1, 'Happy': 1, 'Neutral': 0, 'Sad': -1, 'Surprise': 1}

app = Flask(__name__,template_folder="Template")


# Define the dimensions and frame rate of the video
width = 640
height = 480
fps = 5

# Start the ffmpeg process
command = ['ffmpeg', '-y', '-f', 'avfoundation', '-i', ':0', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-', '-pix_fmt', 'yuv420p', '-f', 'mpegts', 'output.ts']
process = None

# Video capture flag
is_recording = False

# Start the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

# Define the speech recognition object outside of the main block
r = sr.Recognizer()
r.pause_threshold = 2

def exception_handler(exception_type, value, traceback):
    print(value)
    print(traceback)

sys.excepthook = exception_handler

@app.route('/')
def index():
    try:
        # Read the text file into a DataFrame
        df = pd.read_csv('emotions.txt', sep=', ', header=None, names=['Timestamp', 'Emotion'], engine='python')

        # Define the mapping of emotions to numerical values
        emotion_map = {'Angry': -1, 'Disgust': -1, 'Fear': -1, 'Happy': 1, 'Neutral': 0, 'Sad': -1, 'Surprise': 1}

        # Map the emotions to their numerical values
        df['Value'] = df['Emotion'].map(emotion_map)

        # Calculate the cumulative sum of the emotion values
        df['Cumulative Value'] = df['Value'].cumsum()

# Create a line plot of the cumulative values
        plt.plot(df.index, df['Cumulative Value'])

        # Add labels and title to the plot
        plt.xlabel('Row Number')
        plt.ylabel('Cumulative Value')
        plt.title('Cumulative Emotions over Time (Indexed by Row Number)')


        # Save the plot to a PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image in base64 to be able to display it in the HTML template
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Return the HTML template with the plot image
        return render_template('index.html', plot_url=plot_url)
    except Exception as e:
        # Handle the exception
        print('An error occurred:', e)
        return 'An error occurred while generating the cumulative emotions plot', 500

@app.route('/refresh_plot')
def refresh_plot():
    try:
        # Read the text file into a DataFrame
        df = pd.read_csv('emotions.txt', sep=', ', header=None, names=['Timestamp', 'Emotion'], engine='python')

        # Define the mapping of emotions to numerical values
        emotion_map = {'Angry': -1, 'Disgust': -1, 'Fear': -1, 'Happy': 1, 'Neutral': 0, 'Sad': -1, 'Surprise': 1}

        # Map the emotions to their numerical values
        df['Value'] = df['Emotion'].map(emotion_map)

        # Calculate the cumulative sum of the emotion values
        df['Cumulative Value'] = df['Value'].cumsum()

# Create a line plot of the cumulative values
        plt.plot(df.index, df['Cumulative Value'])

        # Add labels and title to the plot
        plt.xlabel('Row Number')
        plt.ylabel('Cumulative Value')
        plt.title('Cumulative Emotions over Time (Indexed by Row Number)')


        # Save the plot to a PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the image in base64 to be able to display it in the HTML template
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Return the plot image as HTML code
        response = make_response('<img id="plot" src="data:image/png;base64,{}" />'.format(plot_url))
        response.headers['Content-Type'] = 'text/html'
        return response
    except Exception as e:
        # Handle the exception
        print('An error occurred:', e)
        return 'An error occurred while refreshing the cumulative emotions plot', 500




def gen_frames():  
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            # Find the largest face
            largest_area = 0
            largest_face = None
            for (x, y, w, h) in faces:
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, w, h)

            if largest_face is not None:
                (x, y, w, h) = largest_face
                # Draw a rectangle around the face with four corners
                line_length = w // 4
                cv2.line(frame, (x, y), (x + line_length, y), (255, 0, 0), 2)
                cv2.line(frame, (x, y), (x, y + line_length), (255, 0, 0), 2)

                cv2.line(frame, (x + w - line_length, y), (x + w, y), (255, 0, 0), 2)
                cv2.line(frame, (x + w, y), (x + w, y + line_length), (255, 0, 0), 2)

                cv2.line(frame, (x, y + h - line_length), (x, y + h), (255, 0, 0), 2)
                cv2.line(frame, (x, y + h), (x + line_length, y + h), (255, 0, 0), 2)

                cv2.line(frame, (x + w - line_length, y + h), (x + w, y + h), (255, 0, 0), 2)
                cv2.line(frame, (x + w, y + h - line_length), (x + w, y + h), (255, 0, 0), 2)

                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    signal, sr = librosa.load('augg4.wav', sr=125)
                    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=40)
                    mfccs = np.expand_dims(mfccs, axis=-1)
                    mfccs = np.expand_dims(mfccs, axis=0)

                    video_prediction = classifier.predict(roi)
                    audio_prediction = audio_classifier.predict(mfccs)  # predict the emotion from the audio signal
                    combined_prediction = 0.5 * video_prediction + 0.5 * audio_prediction  # combine the predictions from both models
                    
                    label = emotion_labels[combined_prediction.argmax()]

                    # draw a rectangle around the text
                    rect_x, rect_y = x, y+h+10
                    rect_w, rect_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    rect_w += 10
                    rect_h += 10
                    cv2.rectangle(frame, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (0, 0, 0), cv2.FILLED)

                    # write the emotion label inside the rectangle
                    text_color = (255, 255, 255) # default to white
                    if label == 'Angry':
                        text_color = (0, 0, 255) # red for angry
                    elif label == 'Happy':
                        text_color = (0, 255, 0) # green for happy
                    elif label == 'Sad':
                        text_color = (255, 0, 0) # blue for sad
                    elif label == 'Surprise':
                        text_color = (255, 255, 0) # yellow for surprise
                    elif label == 'Fear':
                        text_color = (255, 0, 255) # purple for fear
                    elif label == 'Disgust':
                        text_color = (0, 255, 255) # cyan for disgust

                    # Change font size and color
                    font_size = 1
                    font_thickness = 2
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    cv2.putText(frame, label, (rect_x+5, rect_y+rect_h-5), font, font_size, text_color, font_thickness)
            
            # write labels with timestamp to file
            with open('emotions.txt', 'a') as f:
                f.write(f"{strftime('%Y-%m-%d %H:%M:%S')}, {(label)}\n")
            
            # Write the frame to the ffmpeg process
            if is_recording and process is not None and process.poll() is None:
                try:
                    process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    process.stdin.close()
                    process.wait()
                    break

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define a function to update the plot
def update_plot():
    while True:
        # Read the text file into a DataFrame
         with open('emotions.txt', 'r') as f1:
                df = f1.readlines()

                # Map the emotions to their numerical values
                df['Value'] = df['Emotion'].map(emotion_map)

                # Calculate the cumulative sum of the emotion values
                df['Cumulative Value'] = df['Value'].cumsum()

                # Clear the current plot
                plt.clf()

                # Create a line plot of the cumulative values
                plt.plot(df['Timestamp'], df['Cumulative Value'])

                # Add labels and title to the plot
                plt.xlabel('Timestamp')
                plt.ylabel('Cumulative Value')
                plt.title('Cumulative Emotions over Time')

                # Save the plot to a PNG file
                plt.savefig('plot.png', bbox_inches='tight')

                # Wait for 1 second before updating the plot again
                time.sleep(1)

# Start a thread to update the plot in the background
thread = threading.Thread(target=update_plot)
thread.daemon = True
thread.start()

def transcribe_audio():
    with sr.Microphone(device_index=0) as mic:
        while True:
            audio = r.listen(mic, timeout=5, phrase_time_limit=3)
            try:
                transcript = r.recognize_google(audio)
            except sr.UnknownValueError:
                transcript = ""
            app.config['transcript'] = transcript
    
            with open('transcript.txt', 'a') as f:
                 f.write(f"{strftime('%Y-%m-%d %H:%M:%S')}, {(transcript)}\n")

@app.route('/download_report')
def download_report():
    # open the emotions file
    with open('emotions.txt', 'r') as f1:
        data1 = f1.readlines()

    # open the transcript file
    with open('transcript.txt', 'r') as f2:
        data2 = f2.readlines()

    # combine the data into a dictionary using the timestamp as the key
    combined_data = {}
    for line in data1:
        timestamp, *emotions = line.strip().split(',')
        combined_data[timestamp] = {'emotions': emotions}

    for line in data2:
        timestamp, transcript = line.strip().split(',')
        if timestamp in combined_data:
            combined_data[timestamp]['transcript'] = transcript
        else:
            combined_data[timestamp] = {'transcript': transcript}

    # create a list of dictionaries with the combined data
    rows = []
    for timestamp, data in combined_data.items():
        row = {'Timestamp': timestamp}
        if 'emotions' in data:
            emotions = data['emotions']
            for i, emotion in enumerate(emotions):
                row[f'Emotion {i+1}'] = emotion
        if 'transcript' in data:
            row['Transcript'] = data['transcript']
        rows.append(row)
    
    # write the data to a CSV file
    with open('emotions.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Timestamp', 'Emotion 1', 'Transcript','Emotion 2', 'Emotion 3', 'Emotion 4', 'Emotion 5', 'Emotion 6', 'Emotion 7'])
        writer.writeheader()
        writer.writerows(rows)
        # delete the files
        
    os.remove('emotions.txt')
    os.remove('transcript.txt')

    # create a BytesIO object to handle the output in memory
    output = io.BytesIO()
    # write the data to the BytesIO object
    with open('emotions.csv', 'r') as f:
        output.write(f.read().encode('utf-8'))
    # reset the buffer's file pointer to the beginning
    output.seek(0)
    # force browser to download the file as an attachment
    return Response(output.getvalue(),
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=emotions.csv'})

                
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording')
def start_recording():
    global process, is_recording
    is_recording = True
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    return 'Recording started'

@app.route('/stop_recording')
def stop_recording():
    global process, is_recording
    is_recording = False
    process.stdin.close()
    process.wait()
    return 'Recording stopped'

@app.route('/download_video')
def download_video():
    return send_file('output.ts', as_attachment=True)




@app.route('/transcript')
def transcript():
    return Response(str(app.config['transcript']), mimetype='text/plain')

if __name__ == '__main__':
    # Use multithreading to run the video and speech recognition in separate threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(app.run, debug=True, use_reloader=False, port=5002)
        executor.submit(transcribe_audio)

