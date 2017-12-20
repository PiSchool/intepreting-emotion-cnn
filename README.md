# **Interpreting Deep Learning using LIME**

**Pi School of AI Programme**

[//]: # (Image References)
[image0]: ./images/intro.png "intro"
[image1]: ./images/data_kaggle.png "data"
[image2]: ./images/pipeline.png "lime process"
[image3]: ./images/DeepXplore.png "future"
[image4]: ./images/corresponding_classes.png "analysis_classes"
[image5]: ./images/analysis_correct.png "analysis_right"
[image6]: ./images/cfm.png "matrix"
[image7]: ./images/analysis_wrong.png "analysis_wrong"


### Overview 
--- 

![intro][image0]  
_Fig 1. Highlight features (green segments) which contribute to the prediction (LIME tool)_

We have explored a novel solution based on Local Interpretable Model-agnostic Explanations (called LIME) which visually explains a logic behind on the prediction. We used emotion classification as our case study to determine feasibility of interpreting black-box models within this approach. We demonstrate this by applying the interpreter (LIME) to our emotion classifier to highlight key facial features of each emotion type. By observing the results, we can justify how well this interpretation approach and tool performs based on how substantial these facial captions are aligned with human interpretation.  

In this project, we have experimented with LIME tool[1] on the baseline deep learning model of emotion classifier (Xception) [2] using Kaggle face dataset [3]. For our experiments, we re-trained the emotion classifers on the full images (face and background). We also extended LIME tool to run on the deep models with grayscale images.  


### Dependency
---
See ./Scripts/requirements.txt for required packages on the project  

Package installation
```bash
$ pip install -r requirements.txt

```

### Terminal Installation (Linux) 
**Prepare project directories**
```bash
$ git clone git@github.com:PiSchool/intepreting-emotion-cnn.git
$ cd intepreting-emotion-cnn/emoji-project/
$ mkdir -p Biteam Dataset/Kaggle Interpreters
```

**Download XCeption classifier** 
```bash
$ git clone https://github.com/oarriaga/face_classification.git Biteam
```

**Install LIME interpreter and dependencies**
```bash
$ git clone https://github.com/marcotcr/lime.git Interpreters
$ cd Interpreters/
$ python setup.py install
$ cd ..
//update lime_image.py (copy update to lime main folder)
$ cp -nr Scripts/lime_image.py Interpreters/lime/lime_image.py 
```

**Load csv dataset and put in the right folder**
[link](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
```bash
//mov tar file to ./Dataset/Kaggle/ then unzip to the specific folders
$ tar -xzf ./Dataset/Kaggle/fer2013.tar.gz -C ./Dataset/Kaggle/
$ tar -xzf ./Dataset/Kaggle/fer2013.tar.gz -C ./Biteam/datasets/
```

**Make jpeg images from csv**
```bash
//download: https://kaggle2.blob.core.windows.net/forum-message-attachments/179911/6422/gen_record.py
$ wget -O ./Dataset/Kaggle/fer2013/gen_record.py "https://kaggle2.blob.core.windows.net/forum-message-attachments/179911/6422/gen_record.py"
//convert csv to images
$ python ./Dataset/Kaggle/fer2013/gen_record.py
```

**Move sample tranined model**
```bash
$ cp -nr Scripts/fer2013_mini_XCEPTION.hdf5 Biteam/trained_models/emotion_models/fer2013_mini_XCEPTION.hdf5
```

**Usage**
The project has been running on Ubuntu 16.04 LTS
```bash
$ cd Scripts 
$ jupyter notebook lime_interpret_multiple_analysis_cfm.ipynb
```

### The dataset
---

![data][image1]  
_Fig 2. Kaggle dataset: emotions_

Training dataset (22,967 images)   
image size 	   = 48x48 grayscale 1-channel  
number of classes  = 7 emotions  
number of images   = Angry:3,216, Disgust:344, Fear:3,259, Happy:5,741, Sad:3,863, Surprise:2,533, Neutral:4,011 images  



### Emotion classifier: Convolutional Neural Network
---
In this project, we uses convolutional neural network for emotion classification developed by [2] ([github](https://github.com/oarriaga/face_classification)). We have chosen this as the baseline model because it provides a reasonable accuracy with further insights that could be explored using the interpreter. Before we applied the model to the interpreter, we have re-trained the model with full images to confirm the good accuracy that we should expect. We obtained the test accuracy around 66%.
(see  [code](https://github.com/PiSchool/intepreting-emotion-cnn/blob/master/emoji-project/Scripts/train_emotion_classifier.py) )


### Explain the (LIME) explainer
---

![pipeline][image2]  
_Fig 3. LIME processing pipeline_

**“Preprocessing”**   
1- The ground truth image is first segmented into different sections using Quickshift segmentation.   
2- The next step is to generate N data samples by randomly masking out some of the image regions based on the segmentations.   
   This is resulted in a data matrix of "samples x segmentations" where the first row is kept with no mask applied (all 1).   
3- Each sample is weighted based on how much it is different from the original vector (row 1) using  some ‘distance’ function.  

**“Explanation”**   
4- Each data sample (masked/pertubed image) is passed to the classifier (the model being explained e.g. our emotion classifier) for the prediction.   
5- The data instances (binaries) with the corresponding weights (step 3) and the predicted label (step 4- but one label at the time) are then fit to the K-LASSO or Ridge regression classifier to measure the importance of each feature (segmentation in this case).   
6- The final output are the weights, which are representing significance of each segmented feature on the given class.   
7- The positive (support) and negative (against) segments/features are display based on the given thresholding value   
   (e.g. ‘0’ as the separating boundary of being supportive or not).  


### Data selection
---
To capture essential interpretation without losing generality in the data selection process, we have used ‘Softmax probability’ and ‘Confusion matrix’ as yardsticks in picking interesting cases out of the big dataset as follows.  

**1. Softmax probability**  
    	Case 1: Border line prediction + Correction prediction  
	Case 2: Border line prediction + Wrong prediction  
	Case 3: Confident prediction + Correct prediction  
	Case 4: Confident prediction + Wrong prediction  

* _Border line means_: Top two softmax probabilities are different by less than 10%.  
* _Confident prediction means_: The highest probability is more than 80%  


**2. Confusion matrix**  
	Case 5: Correct prediction  
	Case 6: First (foremost) class on the wrong prediction  
  

Jupyter notebooks for the following analysis   
	* Softmax probability (see [codes](https://github.com/PiSchool/intepreting-emotion-cnn/blob/master/emoji-project/Scripts/lime_interpret_multiple_analysis_sfm.ipynb))  
	* Confusion matrix (see [codes](https://github.com/PiSchool/intepreting-emotion-cnn/blob/master/emoji-project/Scripts/lime_interpret_multiple_analysis_cfm.ipynb))  


### Analysis
---

**1. Visualising the features corresponding to EACH class**  

![analysis][image4]  
_Fig 4. Corresponding label analysis: how the classifier thinks of these features (positive/negative)_  

As for example in the figure 4 (1st row), given the ground truth image ‘angry’, we can understand that the classifier uses an open wide mouth as one of the indication to predict ‘angry’ as it highlighted ‘green’ while other hypothesis emotions (disgust, fear, … ) are displayed with ‘red’ on the same feature.   


**2. Visualising the facial features on the CORRECTED prediction**  

![analysis][image5]  
_Fig 5. Right prediction: features which the classifier uses to make the (right) prediction_  

From figure 5, row 1 shows the testing images followed by indicating the positive (green) and negative (red) features in row 2 and highlighted only positive features in row 3 without other segments (gray out areas). Using LIME, we are able to visualize and understand of how the classifier correlates the image features to the prediction.  

**3. WRONG prediction**  

![analysis][image6]  
_Fig 6. Sad Confusion: a majority of the confusion is 'sad'_  

With the confusion matrix (figure 6), we went deeper to analyse '_why_' majority of the confusion was the 'sad' emotion.  


![analysis][image7]  
_Fig 7. Wrong prediction: dark tone influences the 'sad' prediction_  

From figure 7, we can see that the classifier used the dark tone from the background to indicate the sad emotion, which is undesirable. This is a really profound finding. Without the visualization to understand parts of the logic behind the ‘sad’ prediction, it would be really hard and time consuming to find and understand the issue precisely. In the worst case, it is overlooked.  
 

### Summary and Going Further
---

![go][image3]  
_Fig 8. Analysis on a more safety critical application_


Interpreting model has been highlighted as an important aspect to gain trust on deploying any ‘black-box’ models in the real world. As demonstrated in this project, LIME provides the good insights of the emotion classifier which offer insights behind the prediction as well as realizing faults in training the classifier. In addition, the deep hidden issue such as the miss association of dark color tone with ‘sad’ emotion was discovered very quickly with the visualisation. This has saved a lot of time in diagnosing the problem.

LIME tool is an open framework. It can be applied to many deep learning models without any rectifications. We have seen a great benefit of using the LIME tool. It would be really interesting to apply this tool and framework to more critical applications such as self-driving car. On going work in this area has been published in [4]. 

In this project, we focus on the emotion classification with 7 classes. To our knowledge, it raises the question how well the approach perform in a vicinity of larger dataset and classes (e.g. recognition of 500 different objects). Would a local fidelity as in LIME approach still reflect the good interpretation in the global model? This area of work has not been focused in this project and left to be explored in the future. Secondly, with the small number of classes (e.g. 7 emotions), our technique using softmax probability and confusion matrix is sufficient but still exhausting to pick the interesting samples to investigate. With a hundred of classes or more, we need to go deeper in the data selection to capture all possible faults in the model. This makes the process labor intensive. This issue could be addressed in the future research.

### References
---
[1] M. Ribeiro, S. Singh, C. Guestrin, "Why Should I Trust You?": Explaining the Predictions of Any Classifier,  
    Knowledge Discovery and Data Mining (KDD) Conference, 2016, San Francisco, CA, USA   
    (paper: http://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)  
[2] O. Arriaga, P.G.Plöger and M. Valdenegro, “Real-time Convolutional Neural Networks for Emotion and Gender 
    Classification”, Free Software and Open Source Conference (FrOSCon), 2017,  
    Bonn-Rhein-Sieg University of Applied Science  
    (paper: https://arxiv.org/abs/1710.07557)   
[3] Kaggle, “Challenges in representation learning facial expression recognition challenge”, 2013 Competition,   
    https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data  
[4] K. Pei, Y. Cao, J. Yang, S. Jana, "DeepXplore: Automated Whitebox Testing of Deep Learning Systems",  
    Symposium on Operating Systems Principles (SOSP) 2017, Shanghai, China   
    (paper: https://arxiv.org/pdf/1705.06640.pdf)  
 

# Author

This project was developed by [Luke Phairatt](https://github.com/LukePhairatt) during [Pi School's AI programme](http://picampus-school.com/programme/school-of-ai/) in Fall 2017.

![photo of Luke Phairatt](http://picampus-school.com/wp-content/uploads/2017/11/IMG_2150-2-150x150.jpg)






















