<p align="center"> 
  <img src="Template/Emotion_recognisation.gif" alt="img Logo" >
</p>
<h1 align="center"> Emotion Recognition </h1>
<h3 align="center"> This project used two models to predict the emotions which are video and audio model. In both models we are using Convolutional Neural Network (CNN).</h3>  

</br>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 
 The project aims to develop a system that uses computer vision techniques to predict emotions from facial expressions in real-time. This is achieved through the use of convolutional neural network (CNN) models trained on large datasets of labeled images and expressions. The goal is to provide a tool for therapists to better understand their clients' emotional states and tailor their therapy approaches accordingly, with the ultimate goal of improving and accelerating the therapy process. This project has the potential to make a valuable contribution to the field of therapy and mental health.
</p>
<!-- PRE-PROCESSED DATA -->
<h2 id="Problem-statement"> :diamond_shape_with_a_dot_inside: Problem Statement</h2>

<p align="justify"> 

* Traditional methods of assessing emotions in therapy, such as self-reporting or observation, can be subjective and prone to bias.

* This can make it challenging for therapists to accurately understand their clients' emotional states and tailor their treatment approaches accordingly.

* The development of a computer vision system that can accurately predict emotions from facial expressions in real-time could provide therapists with an objective tool to better understand their clients' emotional states and improve the efficacy and efficiency of the therapy process.

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> :fork_and_knife: Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20for-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>
[![made-with-VSCode](https://img.shields.io/badge/Made%20with-VSCode-1f425f.svg)](https://code.visualstudio.com/) <br>

<!--This project is written in Python programming language. <br>-->
The following open source packages are used in this project:
* Numpy
* Pandas
* TensorFlow
* Keras
* Matplotlib
* Sklearn
* pathlib
* Flask
* librosa
* opencv_contrib_python
* pocketsphinx
* scipy
* SpeechRecognition

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> :cactus: Folder Structure</h2>

    Emotion-Recognition-Capston
    .
    ├── Models
    │   ├── Audio_model .ipynb
    │   └── Video_model.ipynb
    ├── Template
    │   ├── Backup_index.html
    │   ├── index.html
    │   └── style.css
    ├── __pycache__
    │   └── streamlit.cpython-310.pyc
    ├── README.md
    ├── app.py
    ├── emotions.txt
    ├── haarcascade_frontalface_default.xml
    ├── requirements.txt
    └── transcript.txt

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
  
 <h2> Video Dataset:Affectnet</h2>
AffectNet is a database of facial expressions and emotions that was collected and made publicly available by the researchers at the Affective Computing. Group at the University of North Texas. The database was created with the goal of addressing the need for a high-quality, diverse dataset for training and evaluating computer vision algorithms in the area of affect analysis. The dataset was created by collecting and annotating facial images from a variety of sources, including publicly available databases and the internet. AffectNet was first introduced in a research paper by Mahmoud Afifi, Abdenour Hadid, and Jean-Luc Dugelay, which was published in the Proceedings of the International Conference on Affective Computing and Intelligent Interaction (ACII) in 2017. It is widely used by researchers in the field of affective computing and has become a benchmark for evaluating the performance of facial expression recognition algorithms.
<br>
The dataset is publicly available for research purposes and can be accessed through the MBZUAI website: http://mohammadmahoor.com/affectnet/
</p>
<p> <h2> Audio Dataset:Speech Emotion Recognition</h2>
Here I am using about a combination of 4 different dataset to get better reach on different kind of voices and recordings.

1. RAVDESS: Contains speech and song recordings from 24 professional actors expressing eight emotions. Includes 7356 files, each lasting about 3-5 seconds.

2. CREMA-D: Contains speech recordings from 91 actors expressing various emotions. Includes 7,442 clips, each lasting 5-7 seconds.

3. SAVEE: Contains speech recordings from four male actors expressing different emotions. Includes 480 clips, each lasting about 3-4 seconds.

4. TESS: Contains speech recordings from 200 actors expressing seven emotions. Includes 2800 recordings, each lasting about 3-5 seconds.

Merging multiple speech emotion recognition datasets can be a good way to increase the diversity and representativeness of the data, and to reduce potential biases in any single dataset. By combining multiple datasets, you can create a larger and more diverse dataset that includes both male and female voices, as well as longer and shorter speech recordings.
<br>
Dataset Source: https://www.kaggle.com/code/cracc97/emotional-speech-recognition/data

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ROADMAP -->
<h2 id="roadmap"> :dart: Roadmap</h2>

<p align="justify"> 
  Initially We had two models: one using VGG with ImageNet for facial recognition and another using CNN for audio recognition. There was a flask app that was genrating emotions by uploading the video onto a webpage.<br>
  The goals of this project include the following:
<br><br>

1. Evaluate the facial recognition model using the FER dataset and testing on affectnet. But the accuracy was not satisfactory for realtime emotion detection, then we developed a new CNN model and run it on Flask in real-time which is providing accurate results.<br>

2. Integrate the audio recognition model with the facial recognition model to predict emotions in real-time and display the best results out of both models on the webpage.<br>

3. Implement a real-time CSV report that logs emotions and timestamps and can be accessed by authorized users.<br>

4. Implement speech recognition using multi-threading to recognize speech in real-time and log it in the CSV report.<br>

5. Create a real-time graph that displays emotions (1 for good, -1 for bad, 0 for neutral) and shows the trend over time. The graph should be interactive and allow users to hover over data points for more details.<br>

6. Implement video recording of the conference with emotion detection and save the video to a file. The video should be accessible and downloadable by authorized users.<br>

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="Re-Create"> :cactus: Re-Create Steps</h2>

Details of scripts and python files are mentioned at the end for quick reference.

Below Steps need to be followed to re-create the results or clone the project.
1. <b>Clone the Git repository:</b> Start by cloning the Git repository that contains your project code to your local machine. You can use the git clone command to do this.

2. <b>Create a virtual environment:</b> Before running the project, create a virtual environment to isolate the required packages and dependencies from your system's Python environment. You can use virtualenv or conda to create the virtual environment.

3. <b>Install the required packages:</b> Once the virtual environment is created, activate it and install the required packages and dependencies by running the command: pip install -r requirements.txt

4. <b>Run the video and audio models:</b> Navigate to the model folder and run the video.py and audio.py Python scripts to train the models and generate the model.h5, model_.h5, and augg4.wav files.
5. <b>Put the files in the main directory:</b> Move the model.h5, model_.h5, and augg4.wav files to the main directory where the app.py script is located.
6. <b>Run the project:</b>

    a. Navigate to the directory where app.py is located.

    b. Run the app.py Python script by executing the  command in the terminal: python app.py
    
    c. Wait for the project to launch. This may take a few seconds depending on your system's performance.

    d. Once the project is launched, you should see a dashboard that displays the following elements:
    
        i. A real-time emotion detection feed that shows the emotions being detected by the video and audio models.

        ii. A report download button that allows you to download a report of the emotions detected by the models.

        iii. A recording button that allows you to record a video and audio feed and detect emotions in real-time.

        iv. A Matplotlib real-time graph that shows the detected emotions in real-time.
     e. Use the dashboard to interact with the project and test its functionality. You can try recording a video and audio feed to see how the emotion detection works, or download a report to see the emotions detected over a longer period of time.

     f. When you're finished, you can stop the project by closing the dashboard or terminating the Python process in the terminal.

    
    

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- REFERENCES -->
<h2 id="references"> :books: References</h2>

1. Alpaydin, E. (2010) Introduction to machine learning. 2nd ed. Cambridge, Mass: MIT Press (Adaptive computation and machine learning).

2. Mehrabi, N. et al. (2022) ‘A Survey on Bias and Fairness in Machine Learning’. arXiv. Available at: http://arxiv.org/abs/1908.09635 (Accessed: 25 January 2023).

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> :scroll: Contributors</h2>

<p>
  :mortar_board: <i>All participants in this project are Student of Post Graduate Diploma in <a href="https://www.stclaircollege.ca/programs/data-analytics-business">Data Analytics for Business</a> <b>@</b> <a href="https://www.stclaircollege.ca/">St Clair College</a></i> <br> <br>
   
  :man_student: <b>Balraj Singh Ojla </b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>BO16@myscc.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/Rajojla">@ma-Rajojla</a> <br>
  
  :man_student: <b>Diksha Diksha</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>DD178@myscc.ca</a> <br>

  :man_student: <b>Deepkiran </b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>DV34@myscc.ca</a> <br>
  
  :man_student: <b>Siddharth </b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>SR147@myscc.ca</a> <br>
  
  :man_student: <b>Akanksha Akanksha </b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>AA446@myscc.ca</a> <br>
  
</p>
<br><br>
<p>
  :mortar_board: <i>Special Thanks to our Professor for guidance along the way</i> <br> <br>
     
  🧑‍🏫 <b>Umair Durrani</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>UDURRANI@stclaircollege.ca</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/durraniu">@ma-durraniu</a> <br>
</p>

<br>
✤ <i>This was the final project for the course DAB402- Capstone (Winter 2023), at <a href="https://www.stclaircollege.ca/">St Clair College</a><i>


