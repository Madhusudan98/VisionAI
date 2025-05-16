from typing import Optional
from ultralytics import YOLO
import numpy as np
import imutils
import cv2
import os
import torch
import sys
from datetime import datetime  # Added for timestamp
import pyresearch

# Configuration parameters
WIDTH = 800
start_status = "Loading..."
shoplifting_status = "Shoplifting"
not_shoplifting_status = "Not Shoplifting"

cls0_rect_color = (255, 255, 255)    # Yellow for not shoplifting (BGR)
cls1_rect_color = (255, 255, 0)      # Green for shoplifting (BGR)
conf_color = (255, 255, 0)         # Cyan for confidence score (BGR)
status_color = (0, 0, 255)         # Red for status (BGR)
header_color = (0, 255, 255)       # Yellow for header (BGR)
footer_color = (0, 255, 255)       # Yellow for footer (BGR)

frame_name = "Shoplifting Detection"
quit_key = 'q'

class ShopliftingDetector:
    """Class to handle shoplifting detection using YOLO model."""
    
    def __init__(self, weights_path: str, input_path: str, output_path: str = "", frames_dir: str = "saved_frames"):
        self.input_path = input_path
        self.output_path = output_path
        self.frames_dir = frames_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(weights_path)
        self.cap = self._initialize_video_capture()
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        
        # Create directory for saving frames if it doesn't exist
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
            print(f"[INFO] Created directory for saving frames: {self.frames_dir}")
        
        print(f"[INFO] Using device: {self.device}")

    def _load_model(self, weights_path: str) -> YOLO:
        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights not found at: {weights_path}")
            model = YOLO(weights_path).to(self.device)
            if not hasattr(model, 'predict'):
                raise AttributeError("Loaded model has no predict method")
            print(f"[INFO] Successfully loaded model from {weights_path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            sys.exit(1)

    def _initialize_video_capture(self) -> cv2.VideoCapture:
        if not os.path.exists(self.input_path):
            print(f"[ERROR] Input video not found at: {self.input_path}")
            sys.exit(1)
        
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video file: {self.input_path}")
            sys.exit(1)
        return cap

    def _draw_detection(self, frame: np.ndarray, box_data: np.ndarray) -> str:
        xywh = np.array(box_data.boxes.xywh.cpu()).astype("int32")
        xyxy = np.array(box_data.boxes.xyxy.cpu()).astype("int32")
        cc_data = np.array(box_data.boxes.data.cpu())
        status = start_status

        for (x1, y1, _, _), (_, _, w, h), (_,_,_,_,conf,clas) in zip(xyxy, xywh, cc_data):
            if clas == 1:  # Shoplifting detected
                self._draw_shoplifting_box(frame, x1, y1, w, h, conf)
                status = shoplifting_status
            elif clas == 0 and conf > 0.8:  # Not shoplifting with high confidence
                self._draw_normal_box(frame, x1, y1, w, h, conf)
                status = not_shoplifting_status
        
        return status

    def _draw_shoplifting_box(self, frame: np.ndarray, x1: int, y1: int, 
                            w: int, h: int, conf: float) -> None:
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), cls1_rect_color, 2)
        center_x = int(x1 + w/2)
        cv2.circle(frame, (center_x, y1), 6, (0, 0, 255), -1)
        conf_text = f"{np.round(conf*100, 2)}%"
        cv2.putText(frame, conf_text, (x1+10, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

    def _draw_normal_box(self, frame: np.ndarray, x1: int, y1: int, 
                        w: int, h: int, conf: float) -> None:
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), cls0_rect_color, 2)
        conf_text = f"{np.round(conf*100, 2)}%"
        cv2.putText(frame, conf_text, (x1+10, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

    def _setup_video_writer(self, frame: np.ndarray) -> None:
        if self.output_path and self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, 25,
                                        (frame.shape[1], frame.shape[0]), True)

    def _draw_header_footer(self, frame: np.ndarray, status: str) -> None:
        # Header
        header_text = f"Shoplifting Detection - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(frame, header_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, header_color, 2)
        
        # Footer
        footer_text = f"Frame: {self.frame_count} | Status: {status}"
        frame_height = frame.shape[0]
        cv2.putText(frame, footer_text, (10, frame_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, footer_color, 2)

    # def _save_frame(self, frame: np.ndarray, status: str) -> None:
        """Save the current frame to the frames directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        status_tag = status.replace(" ", "_").lower()
        filename = f"frame_{self.frame_count:06d}_{status_tag}_{timestamp}.jpg"
        frame_path = os.path.join(self.frames_dir, filename)
        
        cv2.imwrite(frame_path, frame)
        if self.frame_count % 10 == 0:  # Log every 10th frame to avoid console spam
            print(f"[INFO] Saved frame {self.frame_count} to {frame_path}")

    def process_video(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] End of video reached")
                break

            self.frame_count += 1
            frame = imutils.resize(frame, width=WIDTH)

            try:
                result = self.model.predict(frame,conf=0.05)
                if result is None or not result:
                    print(f"[WARNING] Prediction empty for frame {self.frame_count}")
                    continue
            except Exception as e:
                print(f"[ERROR] Prediction failed at frame {self.frame_count}: {str(e)}")
                break

            status = start_status
            if len(result[0].boxes) > 0:
                status = self._draw_detection(frame, result[0])
            
            # Draw header and footer
            self._draw_header_footer(frame, status)
            
            # Status text (kept for backward compatibility)
            cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, status_color, 2)
            
            # Save the current frame to the frames directory
            # self._save_frame(frame, status)
            
            cv2.imshow(frame_name, frame)

            self._setup_video_writer(frame)
            if self.writer:
                self.writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                print("[INFO] Quit key pressed")
                break

    def cleanup(self) -> None:
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Program terminated successfully")

def main():
    # Specify an output path to save the video
    output_video_path = "shoplifting_output.avi"
    # frames_directory = "saved_frames"
    # frame = cv2.imread("saved_frames/frame_000152_loading..._20250516_232713.jpg")
    model = YOLO("last.pt")
    # result = model.predict(frame)
    # print(result)
    
    detector = ShopliftingDetector(
        weights_path="best.pt",
        input_path="demo.mp4",
        output_path=output_video_path,
    )
    # result = model.predict(frame,conf=0.01)
    # print(result)
    # Assuming `result` is a single frame result (not a list of results)
    # for i, res in enumerate(result):
        # print(f"Frame {i}:")
        # cc_data = np.array(res.boxes.data.cpu())
        # print("cc_data",cc_data)
        # class_ids = cc_data[:, 5].astype(int)
        # print("class_ids length",len(class_ids))

        # print("Detected class IDs:")
        # for class_id in class_ids:
        #     label = res.names.get(class_id, "unknown")
        #     print(f"Class ID: {class_id}, Label: {label}")

    detector.process_video()
    detector.cleanup()
    print(f"[INFO] Video saved to {output_video_path}")
    # print(f"[INFO] Frames saved to {frames_directory}")

if __name__ == "__main__":
    main()