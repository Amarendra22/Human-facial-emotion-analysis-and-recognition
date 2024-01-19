# Human-facial emotion analysis and recognitio with CNN
This project will detect 7 types of facial emotion, angry, disgust, fear, happy, neutral, sad, surprise.

### Packages that I have used and need to be installed
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install tensorflow
- pip install pillow
- I have also used HaarCascade to make boundary around the face.

### The FER2013 dataset can be downloaded from the link below
- from the below link, the dataset needs to be put in the data folder under the project directory
- https://www.kaggle.com/msambare/fer2013

### Train Emotion detector
- used face expression images in the FER2013 Dataset
- Run the command --> python TrainEmotionDetector.py ; to train the ML Model

It took around 1hour 45minutes for my laptop with Ryzen 7 processor, 16gb RAM and Nvidia GeForce GTX 1650 GPU.
After Training, we will find the trained model structure and weights are stored in the project directory as:-
emotion_model.json
emotion_model.h5

Now we can copy these two files created, and paste in model folder in the project directory.

### Run your emotion detection test file
Run the command --> python TestEmotionDetector.py ; to see the output. I have made 2 options to view the result(which 
can be seen in TestEmotionDetector.py file,
- cap = cv2.VideoCapture(0) ; this can be used for Human-facial emotion prediction using laptop/pc camera in real time.
- cap = cv2.VideoCapture("C:\\Users\\amrendra\\Downloads\\boy_-_74278 (720p).mp4") ; this can be used for Human-facial 
  emotion prediction using a video from computer's memory. Just the link to the video needs to be pasted in the code.
  
  Link to the video used: https://drive.google.com/file/d/1XmfWlPhj4-2wyRofiqAO_kqNPiOUmVSW/view?usp=sharing ;
  Link of the output video created: https://drive.google.com/file/d/135Vhj6re6WcUQpGEm190A5FREJ4UWqRs/view?usp=sharing ;
