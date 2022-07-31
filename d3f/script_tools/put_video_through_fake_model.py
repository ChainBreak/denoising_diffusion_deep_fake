import cv2
import datetime
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from d3f.lit_module import LitModule
from d3f.script_tools.video_writer_context_manager import VideoWriter

def main():

    args = parse_command_line_arguments()

    RenderFakeVideo(
        args.video_path,
        args.checkpoint_path,
        args.model_a_or_b,
        args.width,
        args.height,
        )

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("video_path",help="Video file you want to fake")
    parser.add_argument("checkpoint_path",help="Model checkpoint path")
    parser.add_argument("model_a_or_b",choices=["a","b"],help="Use model A or B to fake image")
    parser.add_argument("width",help="desired video width")
    parser.add_argument("height",help="desired video height")

    return parser.parse_args()

class RenderFakeVideo():

    def __init__(self,video_path,checkpoint_path,model_a_or_b,image_width,image_height):

        self.video_path = Path(video_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_a_or_b = model_a_or_b
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        self.model = self.load_model_from_checkpoint()

        self.render_real_fake_video()


    def load_model_from_checkpoint(self):
        model = LitModule.load_from_checkpoint(self.checkpoint_path)
        model.cuda()
        model.eval()
        return model

    def render_real_fake_video(self):

        fps = self.get_input_video_properties()

        output_path = str(self.get_output_video_path())

        w = 2*self.image_width
        h = self.image_height
        
        with VideoWriter(output_path,w,h,fps) as video_writer:
            for real_frame in tqdm(self.open_video_as_generator()):
                
                real_frame,fake_frame = self.convert_real_frame_to_fake(real_frame)

                real_and_fake = np.concatenate([real_frame,fake_frame],axis=1)

                video_writer.write(real_and_fake)
                
                cv2.imshow("real_and_fake",real_and_fake)

                cv2.waitKey(1)

    def get_output_video_path(self):

        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%a_%H%M%S")

        output_name = f"{self.video_path.stem}_model_{self.model_a_or_b}_{datetime_str}.mp4"
        output_path = self.video_path.with_name(output_name)
        return output_path

    def get_input_video_properties(self):
        video_path_str = str(self.video_path.resolve())

        video_reader = cv2.VideoCapture(video_path_str)

        fps = video_reader.get(cv2.CAP_PROP_FPS)

        video_reader.release()

        return fps

            
    def open_video_as_generator(self):

        video_path_str = str(self.video_path.resolve())

        video_reader = cv2.VideoCapture(video_path_str)

        while video_reader.isOpened():

            frame_ok, frame = video_reader.read()

            if frame_ok:
                yield frame
            else:
                break

    def convert_real_frame_to_fake(self,real_bgr):
        
        real_bgr = self.crop_image_at_center(real_bgr)

        real_bgr = self.resize_image(real_bgr)
        
        fake_bgr = self.model.predict_fake(real_bgr, self.model_a_or_b)

        return real_bgr,fake_bgr   

    def crop_image_at_center(self,image):
        h,w,c = image.shape

        width_scale = w / self.image_width
        height_scale = h / self.image_height

        scale = min(width_scale,height_scale)

        crop_width = int(self.image_width * scale)
        crop_height = int(self.image_height * scale)

        x1 = (w - crop_width ) // 2
        y1 = (h - crop_height) // 2

        x2 = x1 + crop_width
        y2 = y1 + crop_height

        return image[y1:y2,x1:x2,:]

    def resize_image(self,image):
        return cv2.resize(
            src=image,
            dsize=(self.image_width, self.image_height),
            interpolation=cv2.INTER_CUBIC,
        )

if __name__ == "__main__":
    main()