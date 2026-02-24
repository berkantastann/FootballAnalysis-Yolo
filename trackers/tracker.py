import cv2 as cv 
from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils import get_bbox_center, get_bbox_width
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 3 # GPU varsa batch size'ı artabilir. 
        detections = []
        
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, verbose=False)
            detections += detections_batch 
        return detections
    
    def get_objects_tracks(self, frame,read_from_stub=False,stub_path=None):
        
        if stub_path is not None and read_from_stub and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frame)
        
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
                    
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
 
            
            if stub_path is not None and read_from_stub:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
        
        return tracks
    
    
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        bbox_center_x,_ = get_bbox_center(bbox)
        bbox_width = get_bbox_width(bbox)
        
        cv.ellipse(frame,
            center=(bbox_center_x, y2),
            axes=(int(bbox_width),int(0.35*bbox_width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv.LINE_4
        )
        
        
        return frame
    
    
    def draw_anotation(self,video_frames,tracks):
        output_frames = []
        
        for frame_number,frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_number]
            ball_dict = tracks["ball"][frame_number]
            referee_dict = tracks["referees"][frame_number]
            
            for player_id,player_info in player_dict.items():
                bbox = player_info["bbox"]
                frame = self.draw_ellipse(frame, bbox, (0,255,0),player_id)
                
            for referee_id,referee_info in referee_dict.items():
                bbox = referee_info["bbox"]
                frame = self.draw_ellipse(frame, bbox, (0,0,255),referee_id)
                
            output_frames.append(frame)
            
        return output_frames
    
    
    
      
    