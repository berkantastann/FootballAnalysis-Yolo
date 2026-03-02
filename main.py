from utils import read_video, save_video
from trackers import Tracker
import cv2 as cv


def main():
    video_frames = read_video("input_videos/input.mp4")
    
    tracker = Tracker("models/best.pt")
    
    tracks = tracker.get_objects_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks.pkl")
    
    for track_id,player in tracks["players"][0].items():
        
        bbox = player["bbox"]
        frame = video_frames[0]
        
        cropped_player = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        cv.imwrite(f"output_videos/cropped_player_{track_id}.jpg", cropped_player)
        
        break
        
    
    output_frames = tracker.draw_anotation(video_frames, tracks)

    save_video(output_frames, "output_videos/outputAnotation.mp4")
    
if __name__ == "__main__":
    main()