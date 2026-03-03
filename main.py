from utils import read_video, save_video
from trackers import Tracker
import cv2 as cv
from team_asigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from utils.bbox_utils import measure_distance,get_bbox_center,get_center_of_bbox

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
            team_color = tuple(int(c) for c in team_assigner.team_colors[team_id])
            tracks["players"][frame_num][player_id]["team_color"] = team_color
    
    player_assigner =PlayerBallAssigner()
    
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    team_ball_control= np.array(team_ball_control)
    
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    save_video(output_frames, "output_videos/outputAnotation.mp4")
    
if __name__ == "__main__":
    main()
