
import cv2

class VideoWriter():

    def __init__(self,output_path,w,h,fps):
        self.output_path = output_path
        self.w = w
        self.h = h
        self.fps = fps

    def __enter__(self):
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')

        self.video_writer = cv2.VideoWriter(
            filename = self.output_path,
            fourcc = four_cc, 
            fps =self.fps, 
            frameSize = (self.w, self.h),
        )

        return self.video_writer

    def __exit__(self,*args):
        self.video_writer.release()