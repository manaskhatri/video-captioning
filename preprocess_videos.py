import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import Model

def load_df(path):
    """
    Loads the csv as a dataframe which contains the captions

    path: path of csv file
    """
    df = pd.read_csv(path)

    return df

def preprocess_df(df):
    """
    Select the english captions and extract only the desired columns from the dataframe like 'Name' and 'Description'

    df: dataframe which contains the captions
    """
    df = df[df['Language'] == 'English'].copy()
    df['Name'] = df[['VideoID', 'Start', 'End']].apply(lambda x: x['VideoID'] + '_' + str(x['Start']) + '_' + str(x['End']), axis = 1)
    data = df[['Name', 'Description']]

    return data

def get_final_list(videos_path, data):
    """
    Returns the list of videos which have captions available.

    videos_path: path to the folder which contains the videos
    data: preprocessed dataframe
    """
    videos_list = os.listdir(videos_path)
    num_videos = len(videos_list)

    for i in range(num_videos):
        videos_list[i] = videos_list[i][:-4]

    captioned_videos = set(data['Name'])
    videos = set(videos_list)
    videos_final = list(videos.intersection(captioned_videos)) #These have both video and caption

    return videos_final

def extract_frames(videos_final, source_path, target_path):
    """
    Extract the frames from the videos and store the extracted frames as images in jpg format

    videos_final: list of video names whose frames are to be extracted
    source_path: path to the videos folder
    target_path: path to the target folder where frames are stored
    """
    for video_name in videos_final:
        print("Extracting from", video_name)
        count = 0
        video_captured = cv2.VideoCapture(source_path + video_name + '.avi')
        path = target_path + video_name + '/'
        os.mkdir(path)

        while(video_captured.isOpened()):
            frameId = video_captured.get(1)
            ret, frame = video_captured.read()

            if ret != True:
                break

            if frameId % 10 == 0:
                filename = "frame" + str(count) + ".jpg"
                count += 1
                cv2.imwrite(path + filename, frame)

        video_captured.release()


    print("All frames extracted")

def select_videos(videos_final, frames_path, min_frames):
    """
    Select videos based on the threshold of the min frames

    videos_final: list of videos with captions
    frames_path: path where frames are stored
    min_frames: min threshold for selection
    """
    videos_selected = []

    for video_name in videos_final:
        if len(os.listdir(frames_path + video_name + '/')) >= min_frames:
            videos_selected.append(video_name)

    return videos_selected

def view_frames(video_path):
    """
    View the frames given the video path.
    """
    frames = os.listdir(video_path)
    n = len(frames)
    frames = os.listdir(video_path)
    n = len(frames)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        img = mpimg.imread(video_path+'/frame'+str(i)+'.jpg')
        plt.imshow(img)
        plt.figure()

def load_video_frames(frames_path, videos_selected):
    """
    Loading the frames into numpy array

    frames_path: path to the folder which contains the frames
    videos_selected: list of final video names which meet the min frames threshold
    """
    X = []
    for video_name in videos_selected:
        l = []
        count = 0

        for img_name in os.listdir(frames_path+video_name):
            if count==15:
                break

            img = plt.imread('dataset/msvd_videos/frames/' + video_name +"/"+ img_name)
            img = cv2.resize(img, (224, 224)) #Resize to 224x224
            l.append(img)
            count+=1

        print("Loading for", video_name)
        X.append(l)

    X = np.array(X)
    return X

def extract_features(frames_path, videos_selected):
    """
    Extracting features from the Frames using VGG16 pretrained model. Output is of shape (n, 15, 25088): For n videos and 15 frames for each video

    frames_path: path to the folder which contains the frames
    videos_selected: list of final video names which meet the min frames threshold
    """
    model = VGG16(weights='imagenet', include_top=True)
    feature_extractor = Model(model.input, model.get_layer('fc2').output)

    X = []
    for video_name in videos_selected:
        l = []
        count = 0

        for img_name in os.listdir(frames_path+video_name):
            if count==15:
                break

            img_path = 'dataset/msvd_videos/frames/' + video_name + "/"+img_name
            img = image.load_img(img_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            features = feature_extractor.predict(img_data)
            features = features.flatten()
            l.append(features)
            count+=1

        print("Loading for", video_name)
        X.append(l)

    X = np.array(X)
    return X
