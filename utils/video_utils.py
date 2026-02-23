import cv2 as cv

def read_video(video_path):
    cap = cv.VideoCapture(video_path)
    frames = []
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(frames, output_path):
    height, width, _ = frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, 24.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    
