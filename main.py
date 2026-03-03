from utils import read_video, save_video
from trackers import Tracker
import cv2 as cv
from team_asigner import TeamAssigner

def main():
    video_frames = read_video("input_videos/input.mp4")
    
    tracker = Tracker("models/best.pt")
    
    tracks = tracker.get_objects_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks.pkl")
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    team_assigner = TeamAssigner()
    
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team_id = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team_id
            tracks["players"][frame_num][player_id]["color"] = team_assigner.team_colors[team_id]
    
    
    
    output_frames = tracker.draw_anotation(video_frames, tracks)

    save_video(output_frames, "output_videos/outputAnotation.mp4")
    
if __name__ == "__main__":
    main()