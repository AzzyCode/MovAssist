import cv2
import threading
import time
import logging
import numpy as np

from queue import Queue, Full, Empty
from typing import Callable, Optional, Tuple
from contextlib import contextmanager
from imutils.video import FPS

from utils import rescale_frame


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(
        self, 
        video_path: str, 
        process_frame_callback: Callable, 
        queue_size: int = 10, 
        scale: Optional[float] = None, 
        resize_dim: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize VideoProcessor with video source and processing parameters.
        
        Args:
            video_path: Path to video file or camera index
            process_frame_callback: Function to process each frame
            queue_size: Maximum size of frame buffer
            scale: Scale factor for frame resizing
            resize_dim: Target dimensions for frame resizing
        """
        
        self.video_path = video_path
        self.frame_queue = Queue(maxsize=queue_size)
        self.process_frame_callback = process_frame_callback
        self.stop_flag = threading.Event()
        
        self.video_fps = 60 # Default FPS
        self.scale = scale
        self.resize_dim = resize_dim
        
        self._threads = []


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
        fps = FPS().start()
        frame_count = 0
        
        try:
            while not self.stop_flag.is_set() or self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                frame_count += 1
                elapsed_time = frame_count / self.video_fps
                current_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
                
                if self.scale or self.resize_dim:
                    frame = rescale_frame(frame, scale=self.scale, target_dim = self.resize_dim)
                
                processed_frame = self.process_frame_callback(frame)
                
                if processed_frame is None:
                    continue
                
                fps.update()
                  
                cv2.imshow("Processed Frame", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested stop.")
                    self.stop_flag.set()
                    break
                
                
        except Exception as e:
            logger.error(f"Error in process_frames: {e}")
            self.stop_flag.set()
        finally:
            fps.stop()
            cv2.destroyAllWindows()
            logger.info(f"Elapsed time: {fps.elapsed()} seconds")
            logger.info(f"Average FPS: {fps.fps()}")
                

    def _log_performance_stats(self, fps) -> None:
        """Log performance statistics."""
        logger.info(f"Processing completed:")
        
        
    
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
            
        cv2.destroyAllWindows()
    
            
     