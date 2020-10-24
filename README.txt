DETECTING CORRECT VESTIBULAR REHABILITATION EXERCISES THROUGH 
MACHINE LEARNING USING CONVOLUTIONAL NEURAL NETWORKS
---------------------------------------------------------------
The project is aimed at helping people with vestibular movement disorders get healthy by doing 
exercises as home treatment. In this, using the convolutional neural network, the model is 
trained and the movements on the screen are compared with the model trained, and a result screen 
is returned as a percentage.


Prerequisites
-------------

For Python version:
Choose the version in this link https://www.python.org/downloads/
*Python 3.7


For PyCharm Editor:
Choose the community in this link and follow the installation 
steps; https://www.jetbrains.com/pycharm/download/#section=windows
*PyCharm Editor(depends on users)


In PyCharm Editor;
Install the libraries at File -> Setting -> Project:(with name) -> Project Interpreter 
-> Click "+" to add them -> Write Library Name
*Keras (v2.3.1) library, 
*glob (v0.7) library, 
*numpy (v1.18.1) library, 
*cv2 (v4.1.2.30) library, 
*os library, 
*playsound(v1.2.2) library, 
*PyQt5 (v5.14.1) library, 
*time library, 
*sys library.


Computer Preporties
-------------------
*A laptop 
*A camera (640*480 Pixel), 
*4GB RAM, 
*i5 4210n Processor, 
*2.6GHz Processor speed, 
*Win10 x64 Operating System 


appendix.doc
------------
Here includes the codes and their descriptions beside them. 50 trials in the program and accordingly 
Test No, Movement Number, Correct Move Time (seconds), Incorrect Move Time (seconds), 
Expected Correct Percentage, Expected Incorrect Percentage, Obtained Correct Percentage, 
Obtained Incorrect Percentage are included in appendix 1.


prediction.py
-------------
here, the model with the previously trained .h5 extension is included in the program and 
by opening the camera, every image taken from the camera will be calculated correctly or 
incorrectly for 30 seconds, and a correct or incorrect percentage will be given on the result screen.
You can see the codes in Appendix 2.


model_training.py
-----------------
Here, the model was trained and the trained model was recorded in a file with the extension .h5.
A total of 720 training images and 120 test images were used for the 1b and 2a movements for training.
You can see the codes in Appendix 3.


Authors
-------
Aydýn TANRIVERDÝ
Görkem Genar AKKAYA
Sümeyye Nur ÖZGENÇ
Buse HACIÞABANOÐLU
Melek Baþak ÖZKAN


On the ubis system could not be put here the images because of the oversize when uploading 
the trained model.