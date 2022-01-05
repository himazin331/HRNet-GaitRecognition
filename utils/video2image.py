import os
import sys
import cv2

# Convert video to image
class Video2Image():
    def __init__(self):
        self.frame_list = []

    def convert(self, video_path):
        video = cv2.VideoCapture(video_path) # 動画読み込み
        if video.isOpened():
            while True:
                ret, frame = video.read()
                if ret:
                    self.frame_list.append(frame) # フレーム
                else:
                    break
        else:
            raise Exception("Failed to load the video.")

        return self.frame_list