from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video("input_videos/input.mp4")
    
    tracker = Tracker("models/best.pt")
    
    tracks = tracker.get_objects_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks.pkl")
    
    #save_video(video_frames, "output_videos/output.mp4")
    
if __name__ == "__main__":
    main()