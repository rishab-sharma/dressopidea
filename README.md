# Dressopedia - Deep Learning App

Following is my Repo for the Deep Learning App for dress classification

## Approach

I took two approach different approach for the above task , in order to get the best result possible.

1. Fine Tuning

In this approach I took a InceptionV3 model and changed the output layers in order to prepare it for 12 class prediction.

Though in the task description , it was said to have 13 classes , I found only 12 non-empty in my zip.

Then I freezed the input and top layers of this base model and trained the new added layers , on the given data set.

For this I have to resize the images down to 299 x 299 pixels.

After this phase I freeze the bottom layers and recompiled the model with SGD optimiser and Fine Tune the base model.

Thus in this approach I took a InceptionV3 model because , this model , having being trained on imagenet , have a good deal knowledge of edge detection and feature extraction and classification. Therefore this approach was to achieve a state of the art result.

2. Custom CNN

In this approach I Designed a CNN 13 conv2D layers and Maxpolling 3D layers. At the end of CNN  flattened the output , connected it to 3 Dense layers along with a softmax end.

This approach was for designing a custom model for the sole purpose of classifying clothes.

## Preprocessing 

I preprocessed the data using pandas, keras Image data generator , openCV.

## Training

I Only trained the model for a 3-4 epochs due to time and resource constraint, and saved it using checkpointer.

Training loss was about = 1.99
Training accuracy was about = 3.04 

## Conclusion 

Thus I did this task with my most dedication and best of my knowledge. The model result may not be too correct because of less training , as I just got free from my exams and I did the best I can do.

Thank you for your time and Consideration.
It was a great learning experience.
