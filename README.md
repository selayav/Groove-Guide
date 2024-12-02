# Groove-Guide: Automatic Music Genre Classification with Explainable AI
## Introduction
Groove-Guide aims to classify the genres of music audios. This problem is interesting to me because I am a pianist, and I wanted to explore the applictions of machine learning to music.

Different genres have different features, including timbre, harmony and dynamics, but the genres can also have unique time dependencies of these features. Groove-Guide uses a hybrid CNN-LSTM model to classify music audios into genres. This hybrid model will be able to extract features of the audio, using the CNN architecture, and then learn time dependencies using the LSTM architecutre. However, one limitation of deep learning algorithms is that they are black-box algorithms. That is, they are not interpretable. To overcome this limitation, Groove-Guide uses SHapley Additive exPlanations (SHAP) values to identify regions in the audio that were most useful for the classification.

## Set-Up
The dataset used for this task is the GTZAN dataset [1]. This dataset contains music of 10 genres (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock). Each genre has 100 audio files that are 30 seconds in length. To avoid using too many computation resources, my plan was to use Music Genre Classification Using Convolutional Neural Networks by Christopher R. Landschoot [2] as a pretrained model, and then add the LSTM architecture. However, the CNN model trained by Landschoot was not compatible with the version of the dataset at hand. So, instead, I processed the data and trained, tested and validated the CNN model created by Landschoot myself. I then added the LSTM architecture and compared the results of the CNN-LSTM model to the CNN model.

Audio files can be represented as Spectrograms, which shows which frequencies are activated at a specific time. The three heatmaps below were obtained from [2]. 

<img width="773" alt="image" src="https://github.com/user-attachments/assets/db76e52e-bd3b-4867-ac37-10dea17c7019">

To filter this data to frequencies to align more with the range of human hearing, it is common to scale spectrogram values using the Mel-Scale. This uses 128 Mel Frequency Cepstral Coefficients (MFCCs).

<img width="773" alt="image" src="https://github.com/user-attachments/assets/36965dc9-1b48-462f-a9f4-3813a46ce535">

Finally, to reduce the dimensionality of the data, the MFCCs can be summarized with 13 coefficients instead of 128. To increase the dataset, each 30 second audio was split into ten 3 second audios.

<img width="793" alt="image" src="https://github.com/user-attachments/assets/16d553cf-56bf-49c9-ab87-7db28f85fcdc">

To train the CNN model, each 3 second clip was flipped to create a new audio clip. This further augmented the dataset. The table below shows the CNN architecture. Unlike Landschoot, I incorporated early stopping to limit the computational resources used.

<img width="638" alt="Screenshot 2024-12-02 at 3 01 35 PM" src="https://github.com/user-attachments/assets/2d3d24e5-8e85-4475-a4ed-44f087e10c33">

Once the CNN model was trained, tested and validated, I added layers to incorporate the LSTM model to the pretrained CNN model, following which I trained the same dataset on the CNN-LSTM model. The table below shows the CNN-LSTM architecture

<img width="480" alt="Screenshot 2024-12-02 at 3 02 58 PM" src="https://github.com/user-attachments/assets/4b2dd1d8-90dd-4b62-a82b-b0667c3e09ec">

The models were trained using the T4 GPU on Google Colab. 

Following this, SHAP values were obtained and plotted over the heatmap to identify features that are useful for classifying each audio clipping.

## Results
Overall, it was observed that the CNN-LSTM model outperformed the CNN model. The CNN-LSTM model was trained within fewer epochs and achieved higher accuracies and lower losses. 

<img width="948" alt="image" src="https://github.com/user-attachments/assets/b23f3369-744d-4f4b-9532-1ba88d213d07">

The CNN-LSTM model achieved a test accuracy of 0.85 and a test loss of 0.56, while the CNN model achieved an accuracy of 0.82 and test loss of 0.65. 

<img width="734" alt="image" src="https://github.com/user-attachments/assets/407ce174-08db-4778-9ee8-2292b6f85085">

Furthermore, it was observed the CNN-LSTM model was able to identify more features for classification.

<img width="941" alt="image" src="https://github.com/user-attachments/assets/32985900-1ffa-4dd0-a3fb-a4f49c9dbb1e">

## Discussion
As seen, the CNN-LSTM model produced better results than a CNN model alone. This is expected since music is time dependent; features change in predictable ways as time goes on, and features in different genres evolve in different ways. Unlike other models, I inorporated Groove-Guide allows users to explore SHAP values of features, allowing for model interpretability. This invites further exploration by reverse engineering only the important features back into an audio format, enabling a auditory insight into what makes a particular genre distinct.

## Conclusion
In this project, I created an explainable CNN-LSTM model to classify music genres using the GTZAN dataset. This model achieved a test accuracy of 0.85. Using SHAP values, it is also possible to identify what features in the audio contribute to the classification.

## Citations
[1] https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

[2] https://github.com/crlandsc/Music-Genre-Classification-Using-Convolutional-Neural-Networks/tree/main
