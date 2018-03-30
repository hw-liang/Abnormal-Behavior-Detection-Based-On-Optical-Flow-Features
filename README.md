# Abnormal-Behavior-Detection-Based-On-Optical-Flow-Features
This project applied computer vision and machine learning methods to detect abnormal behavior in crowd.(Finished on Dec 19, 2017)

The goal of our work is to establish a video system that can distinguish abnormal behaving people from normal ones. We focused on a privacy-preserving video system and obtained the internal information of video with image processing techniques. Using optical flow features, we estimated and segmented inhomogeneous crowds composed of pedestrians that travel in different directions. With these features, we applied several classifying algorithms to tell the normal behaviour from the abnormal. In this process, we used grid search method to find the best hyper-parameter. We validated both the crowd segmentation algorithm, and the crowd counting system, on a large pedestrian dataset (200 frames of video, containing 4,988 total pedestrian instances). This project could help monitoring abnormal behaved objects in public environment with vision information.

- Language: Python3.6.1
- Dependencies: OpenCV, Scikit-Learn, Scikit-Image, and Numpy (Anaconda recommanded)
- Feature extraction: Morphological filter, Normalisation, Optical Flow
- Classification Model: SVM, LogisticRegression, KNN
