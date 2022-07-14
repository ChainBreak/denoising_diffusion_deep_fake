import cv2
from tqdm import tqdm
import argparse
from pathlib import Path



def main():

    args = parse_command_line_arguments()

    VideoToImages(
        args.video_path,
        args.width,
        args.height,
        )

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("video_path",help="path to the config yaml")
    parser.add_argument("width",help="output image width")
    parser.add_argument("height",help="output image height")

    return parser.parse_args()

class VideoToImages():

    def __init__(self,video_path,image_width,image_height):

        self.video_path = Path(video_path)
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        self.create_output_folder()

        self.convert_video_to_images()

    def create_output_folder(self):

        output_dir_name = f"{self.video_path.stem}_w{self.image_width}_h{self.image_height}"
        self.output_dir_path = self.video_path.parent / output_dir_name
        self.output_dir_path.mkdir(exist_ok=True)
    
    def convert_video_to_images(self):

        for frame_index, frame in tqdm(enumerate(self.open_video_as_generator())):

            self.resize_and_save_frame_as_image(frame,frame_index)

    def open_video_as_generator(self):

        video_path_str = str(self.video_path.resolve())

        video_reader = cv2.VideoCapture(video_path_str)

        while video_reader.isOpened():

            frame_ok, frame = video_reader.read()

            if frame_ok:
                yield frame
            else:
                break

    def resize_and_save_frame_as_image(self,frame,frame_index):

        frame = self.crop_image_at_center(frame)

        frame = self.resize_image(frame)

        self.save_frame_to_disk(frame,frame_index)

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

    def save_frame_to_disk(self,frame,frame_index):

        file_path = str( self.output_dir_path / f"{frame_index:05}.jpg" )

        cv2.imwrite(file_path,frame)



if __name__ == "__main__":
    main()