import cv2
import threading
import logging
import os
from datetime import datetime
from queue import Queue, Full, Empty
from typing import Callable, Optional, Tuple
from contextlib import contextmanager
#from imutils.video import FPS

from src.utils import preprocess_frame
from src.core import Squat, Pushup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(
        self, 
        video_path: str, 
        process_frame_callback: Callable, 
        queue_size: int = 40, 
        scale: Optional[float] = None, 
        resize_dim: Optional[Tuple[int, int]] = None,
        squat_instance: Squat = None,
        pushup_instance: Pushup = None,
        save_replay: bool = False 
    ):
        """
        Initialize VideoProcessor with video source and processing parameters.
        """
            
        self.video_path = video_path
        self.frame_queue = Queue(maxsize=queue_size)
        self.process_frame_callback = process_frame_callback
        self.stop_flag = threading.Event()
        
        self.video_fps = 60 # Default FPS
        self.scale = scale
        self.resize_dim = resize_dim
        
        self._threads = []
        
        self.squat = squat_instance
        self.pushup = pushup_instance
        
        self.save_replay = save_replay
        self.video_writer = None

    @contextmanager
    def _video_capture(self):
        """Context manager for handling video capture resource."""
        cap = cv2.VideoCapture(self.video_path)
        try: 
            if not cap.isOpened():
                raise RuntimeError(f"Failer to open video source: {self.video_path}")
            yield cap
        finally:
            cap.release()        
            
    
    def _init_video_properties(self, cap: cv2.VideoCapture) -> None:
        """Initialize video properties from capture object."""
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or self.video_fps
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"FPS: {self.video_fps},"
                    f"Total frames: {self.total_frames},"
                    f"Resolution: {self.frame_width}x{self.frame_height}")
     
        if self.resize_dim:
            self.frame_width, self.frame_height = self.resize_dim
            
        if self.video_fps == 0 or self.video_fps is None:
            self.video_fps = 30 
     
        if self.save_replay:  
            os.makedirs("replays", exist_ok=True)
            exercise_name = "exercise"
            if self.squat:
                exercise_name = "squat"
            elif self.pushup:
                exercise_name = "pushup"
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_file = os.path.join("replays", f"{exercise_name}_{timestamp}.avi")            
            
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            frame_size = (self.frame_width, self.frame_height)
            self.video_writer = cv2.VideoWriter(output_file, fourcc, self.video_fps, frame_size)  
           
            
    def read_frames(self):
        """Read frames from video source and add to queue."""
        try:
            with self._video_capture() as cap:
                self._init_video_properties(cap)
                
                while not self.stop_flag.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("End of video stream reached.")
                        break

                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                    except Full:
                        logger.warning("Frame queue is full. Skipping frame.")
                        continue
        
        except Exception as e:
            logger.error(f"Error in read_frames: {str(e)}")        
            self.stop_flag.set()
        finally:
            self.stop_flag.set()
            
            
    def process_frames(self):
        """Process frames from queue and display results."""
        #fps = FPS().start()
        
        try:
            while not self.stop_flag.is_set() or self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except Empty:
                    continue
                
                
                processed_frame = preprocess_frame(frame, target_height=768, target_width=1024)
                processed_frame = self.process_frame_callback(processed_frame)
                
                if processed_frame is None:
                    continue
                
                #fps.update()
                  
                if self.save_replay and self.video_writer is not None:
                    if processed_frame.shape[1] != self.frame_width or processed_frame.shape[0] != self.frame_height:
                        processed_frame = cv2.resize(processed_frame, (self.frame_width, self.frame_height))
                    self.video_writer.write(processed_frame)    
                
                cv2.imshow("Processed Frame", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested stop.")
                    self.stop_flag.set()
                    break
                
        except Exception as e:
            logger.error(f"Error in process_frames: {e}")
            self.stop_flag.set()
        finally:
            #fps.stop()
            cv2.destroyAllWindows()
            #logger.info(f"Elapsed time: {fps.elapsed()} seconds")
            #logger.info(f"Average FPS: {fps.fps()}")

            if self.video_writer is not None:
                self.video_writer.release()
                logger.info("VideoWriter released. Video filed finalized.")
                        
        
    def start(self):
        """Start video processing."""
        try:
            reader_thread = threading.Thread(target=self.read_frames)
            processor_thread = threading.Thread(target=self.process_frames)
            
            self._threads = [reader_thread, processor_thread]
            
            for thread in self._threads:
                thread.start()
                
            for thread in self._threads:
                thread.join()
        
        except Exception as e:
            logger.error(f"Error in start: {str(e)}")
            self.stop_flag.set()
        finally:
            self.cleanup()
            
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_flag.set()
        
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        if self.squat:
            self.squat.generate_summary()
        elif self.pushup:
            self.pushup.generate_summary()
            
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info("VideoWriter released.")
            
        cv2.destroyAllWindows()
    
            
     