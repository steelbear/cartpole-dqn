from moviepy.editor import VideoFileClip, concatenate_videoclips
import os


def concatVideos(result_name, path):
    currentVideo = None
    fileList = os.listdir(path)
    fileList.sort()
    for filePath in fileList:
        if filePath.endswith(".mp4"):
            if currentVideo == None:
                currentVideo = VideoFileClip(path + filePath)
                continue
            video_2 = VideoFileClip(path + filePath)
            currentVideo = concatenate_videoclips([currentVideo, video_2])
    currentVideo.write_videofile(result_name + ".mp4")

