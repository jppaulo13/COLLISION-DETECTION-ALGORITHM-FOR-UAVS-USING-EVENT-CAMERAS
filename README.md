# COLLISION-DETECTION-ALGORITHM-FOR-UAVS-USING-EVENT-CAMERAS

This repository contains an implementation of a collision detection algorithm for UAVs using event cameras and the data of 9 recorded sequences, includes the events and the imu information. The algorithm takes as input the stream of events from the camera and outputs the locations of the dynamic objeccts within the scene.

Access data through this link: https://drive.google.com/drive/folders/1lhNRML8HVNRZbxZB26XQtKz80BL1BURB?usp=share_link

Prerequisites
Python 3.x
Numpy

Getting Started
To get started, clone the repository to your local machine using the following command:
git clone https://github.com/<username>/collision-detection-uavs-event-cameras.git

Next, install the required packages using the following command:
pip install -r requirements.txt

Running the Algorithm
To run the algorithm, simply run the collision_detection.py file as follows:
python collision_detection.py

Acknowledgments
This project was inspired by the works of A. Mitrokhin and Falanga on event-based collision detection.
  
A. Mitrokhin, C. Fermüller, C. Parameshwara and Y. Aloimonos, "Event-Based Moving Object Detection and Tracking," 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, Spain, 2018, pp. 1-9, doi: 10.1109/IROS.2018.8593805.
  
Falanga, Davide & Kleber, Kevin & Scaramuzza, Davide. (2020). Dynamic obstacle avoidance for quadrotors with event cameras. Science Robotics. 5. eaaz9712. 10.1126/scirobotics.aaz9712. 
