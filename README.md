# Computer-Vision-Project
Developed a computer vision system to automatically detect photophobia by analyzing multimodal facial (expressions) and eye movement data. Utilized a CNN to process video frames capturing pupil dilation, blinking, and squinting under different lighting conditions for real-time discomfort detection.


Automated Detection of Photophobia Using Computer Vision and Multimodal Facial and Eye Movement Analysis

 

1. Project Abstract 

The goal of this research is to develop a real-time computer vision system that can identify photophobia (light sensitivity) by examining eye movements, specifically pupil dilation and blinking patterns. For eye tracking analysis, the data will come from the Labeled Pupils in the Wild (LPW) dataset. This dataset provides high-resolution videos of pupils captured in different light settings. We utilize convolutional neural networks (CNNs) to leverage this dataset to create an adaptive system, that can precisely track these reactions and help with photophobia diagnosis and treatment. This system has potential applications in diagnosing neurological disorders such as migraines, autism, and PTSD. 

2. Introduction 

Photophobia or light sensitivity, manifests as discomfort or pain in response to light. It is often a symptom associated with conditions such as migraines, autism, PTSD, or other eye related diseases. The symptoms (pupil dilation, squinting, and blinking in response to light) are not only typically minor and may vary between individuals but they also overlap with other health reasons. This makes it difficult to accurately diagnose photophobia in real-time. Because light conditions can vary, environmental factors like interior vs outdoor illumination further complicate the detecting procedure. These difficulties may be overcome with a system that could precisely monitor blinking patterns and pupil dilations under different variations in light in real-time, improving the identification and treatment of light sensitivity disorders. Additionally, it can open the door for personalized lighting schemes and environmentally adaptive settings. 

3. Background 

Computer vision for eye diseases detection using pre‐trained deep learning 

techniques and raspberry Pi: This paper describes an automated eye disease detection system that applies pre-trained deep learning models such as InceptionResNetV2 to real-time image processing. It detects problems including cataracts and glaucoma with an accuracy of up to 93 percent. Similar machine learning approaches can be used to improve our photophobia detecting systems. Implementing and comparing different models will increase the accuracy in detecting small eye changes such as pupil response. 

 

Photophobia in neurologic disorders: The study examines the epidemiology and pathophysiology of photophobia in neurological illnesses, focusing on its ambiguous character and numerous nature, which range from primary eye conditions to central nervous system (CNS) and psychiatric disorders. It explores diagnostic and therapy options, focusing on the complex interaction of light-induced sensory impairments. This supports our technique of combining eye-tracking data with facial emotions. 

 

Current understanding of photophobia, visual networks, and headaches: This research explores the clinical and pre-clinical mechanisms that relate visual pathways and photophobia, especially in headache diseases such as migraines. It demonstrates how affected individuals process light differently via image-forming (cone/rod-mediated) and non-image-forming (melanopsin-mediated) pathways. This causes headaches and visual discomfort to worsen when exposed to light. Understanding these networks and their neurological roles improves our project's approach to detecting small pupil and face changes. It supports our approach of employing advanced eye-tracking and machine learning to understand slight changes in varied lighting settings, which improves diagnosis accuracy in light-sensitive. 

 

 

4. Dataset. 

The Labeled Pupils in the Wild (LPW) dataset serves as the primary source of data. It comprises of 22 participants, each recorded in 3 videos under varying natural lighting conditions. The videos capture eye movements, pupil dilation, and blinking patterns in high resolution. This dataset provides ground truth annotations for pupil center coordinates and diameter, which are crucial for training and validating the model. The diversity of lighting conditions and participant demographics in the LPW dataset ensures robustness in real-world scenarios.  

 

5. Method. 

This project involves several key steps, starting with data preprocessing. Frames are extracted from the LPW dataset videos at consistent intervals using OpenCV. These frames are then converted to grayscale to reduce computational complexity and then resized to a standard resolution of 128x128 pixels for uniformity. The dataset also includes annotations such as pupil positions and blink labels, that are crucial for training and evaluation. After preprocessing, the features are extracted using a pre-trained ResNet-18 model. This CNN is particularly effective at capturing spatial features like pupil shape, movement, and dilation patterns. To ensure these features are meaningful, pooling layers are applied to reduce dimensionality while retaining the most valuable information. The extracted features are thereafter used to train a classifier for detecting blinking and a regression model to estimate pupil size and dilation patterns. 

 

5.1.  Network Architecture. 

The CNN architecture is designed to effectively process the preprocessed frames and extract the necessary data. The model takes grayscale frames resized to 128x128 pixels as input. The architecture includes two convolutional layers with 32 and 64 filters, each using a 3x3 kernel and ReLU activation. These layers are followed by max-pooling layers, which down sample the feature maps by a factor of 2, making the model both efficient and robust to noise. Fully connected layers further process the extracted features, with a dense layer containing 128 neurons followed by an output layer. The softmax activation function in the output layer is used for classification, while regression tasks utilize additional dense layers. The model is trained using the Adam optimizer, with cross-entropy loss applied for classification tasks and Mean Squared Error (MSE) for regression. 

 

5.2. Photogrammetry Software 

OpenCV is the photogrammetry software for this project. It is used extensively for preprocessing tasks such as frame extraction, resizing, and grayscale conversion. OpenCV's capabilities for brightness normalization and contrast adjustment ensure that the model can handle variations in lighting conditions well enough. This is vital given the diverse lighting environments present in the LPW dataset. In addition to this, OpenCV supports pupil segmentation and feature extraction. 

5.3. Calibration techniques 

To align pupil measurements across different participants and videos, a calibration technique is employed. This involves normalizing pupil sizes to account for differences in camera distance and perspective. Histogram equalization is used to minimize the effects of lighting variations across frames, ensuring consistent image quality. To further refine the input data, affine transformations are used to correct for slight rotations or distortions in the video frames, aligning the pupil region consistently in every frame. 

6. Experimental Design 

This is structured to evaluate the model's accuracy, precision, and recall in detecting blinking and estimating pupil dilation. The dataset is split into training (70%), validating (20%), and testing (10%). The training set is used to train the model, while the validation set helps monitor performance and fine-tune hyperparameters. The test set is reserved for final evaluation to measure how well the model generalizes to unseen data. The model's performance is assessed using several metrics, including the classification accuracy for blink detection, the Mean Absolute Error (MAE) for pupil size predictions, and the F1 score to evaluate the balance between precision and recall. To test robustness, experiments include evaluations under varying lighting conditions and on unseen participants. The implementation relies on Python libraries such as PyTorch and OpenCV, with high-performance GPUs for training the CNN. 

7. References 

[1] (PDF) computer vision for eye disease detection using pre‐trained deep learning       techniques and Raspberry Pi. (n.d.). https://www.researchgate.net/publication/381884593_Computer_vision_for_eye_diseases_detection_using_pre-trained_deep_learning_techniques_and_raspberry_Pi . 

[2] Wu, Y., & Hallett, M. (2017, September 20). Photophobia in neurologic disorders - translational neurodegeneration. BioMed Central. https://translationalneurodegeneration.biomedcentral.com/articles/10.1186/s40035-017-0095-3  

[3] Noseda, R., Copenhagen, D., & Burstein, R. (2019, November). Current understanding of photophobia, visual networks, and headaches. Cephalalgia: an international journal of headaches. https://pmc.ncbi.nlm.nih.gov/articles/PMC6461529/#S6 . 

          [4] Labeled pupils in the wild: A dataset for studying pupil detection in unconstrained  	environment   Marc Tonsen, Xucong Zhang, Yusuke Sugano, Andreas Bulling. Proc. ACM 	International Symposium on Eye Tracking Research and Applications (ETRA), pp.    139-     	142, 2016. 
