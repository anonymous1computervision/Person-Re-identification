## Introduction

This is a project for detecting and tracking people in a video and making a list of all unique people that have appeared in the video. If a person appears for some frames and then leaves the video for some time and then reappears later, the model should recognize it's the same person and not count this person for a second time.
I have implemented 2 approaches: Tracking and Re-identification.

![](https://cdn-images-1.medium.com/max/1200/1*-WkySYuR7koWY3g_Ikec2A.gif)

[Additional documentation](https://docs.google.com/document/d/1SKYwn4i9t2PlAisoMi7N8Fb6EAc73WSKt04jQUmlZYQ/edit?usp=sharing)

## Approach 1: Re-identification

This is the code `baseline.py`.
I have combined the model described in the paper  [An Improved Deep Learning Architecture for Person Re-Identification](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf) with the Tensorflow object detection API to perform person re-identification in videos.

The model proposed in the paper uses a CNN to perform person re-identification in static images. Given 2 images, the model predicts whether they are of the same person or not. I have used the Tensorflow object detection API to adapt this model for videos.
My code processes each frame independently. It runs the Tensorflow object detector on each frame and obtains the bounding boxes for people in that frame. Using the obtained bounding boxes, the people are cropped out and sent to the person re-identification model. The people in the current frame are compared pairwise with all the previously detected unique people. This determines whether the person was previously detected or not. If the person is not matched with any of the previous people, then he/she is added to the list as a new unique person. His/her image is also added to the database of previously detected people so people in the future frames can be compared with this person.
 
The person re-id model is in `run.py`. This model is used by the main code in `baseline.py` for each pair of images.
 

## Approach 2: Tracking

This is the code `human_detect_track.py`

I implemented a tracking mechanism using intersection-over-union. Previously, the model would iterate over every frame. In each frame, the object detection module would return bounding boxes for the people. And each detected person in each frame was compared with all the previously detected people. This led to a large number of comparisons. 

In this version, the model iterates over all frames and in each frame it obtains the person bounding boxes. The model stores a list of bounding box positions from the previous k(=5 to 10) frames. It matches the bounding boxes from the current frame to the previous frame boxes that are very close in position. In other words, if the current bounding box is very close to a bounding box from the k previous frames- that means that it’s the same person who has just slightly moved between frames.

To determine that 2 bounding boxes from consecutive frames are very close, it uses intersection-over-union.

IOU between 2 boxes   =   Area of the intersection of boxes / Area of the union of boxes

IOU is a measure of the overlap between 2 boxes. If the boxes overlap a lot, IOU is close to 1. If the 2 boxes don't overlap at all, IOU is 0.

So, for each bounding box from the current frame, the model tries to find a bounding box from the k previous frames which greatly overlaps with it (IOU > 0.9). If such a box is found, the model assigns the previous box’s person to the new bounding box. In this way, people are identified without actually running the neural network on them. If a bounding box is not able to match with any previous box, it means that this person just entered the frame and was not there in the previous frame. 

## Approach 3: Tracking + Re-identification

This is the code `human_detect.py`

This model uses both tracking and re-identification. The people are tracked using intersection-over-union between frames. However, in the case that a person enters the video and was not present in the previous frame or the tracking failed for some reason, then the re-identification model in `run.py` is run to determine who this person is and whether he is a new person or was seen before.

In this version, the model iterates over all frames and in each frame it obtains the person bounding boxes. But it doesn’t re-identify each person, only the uncertain ones. The model stores a list of bounding box positions from the previous frame. It matches the bounding boxes from the current frame to the previous frame boxes that are very close in position. In other words, if the current bounding box is very close to a bounding box from a previous frame- that means that it’s the same person who has just slightly moved between frames. This way the model identifies each person only once (when they’re first seen) using the neural network. For the next frames, it just tracks the person.

So, for each bounding box from the current frame, the model tries to find a bounding box from the previous frame which greatly overlaps with it (IOU > 0.9). If such a box is found, the model assigns the previous box’s person to the new bounding box. In this way, people are identified without actually running the neural network on them. If a bounding box is not able to match with any previous box, it means that this person just entered the frame and was not there in the previous frame. In this case, the neural network is run on the person to determine if he has appeared before or if he is a new person totally, in which case he is added to the list of unique people detected. This new person is compared with all the other previously detected people. If a match is found, then that means the person had appeared before but then disappeared for an intermediate period. If a match is not found, that means the person is appearing in the video for the first time and needs to be added to the list of unique people.

