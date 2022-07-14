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
        
        file_path_list = []

        for frame_index, frame in tqdm(enumerate(self.open_video_as_generator())):
            
            file_path =  self.resize_and_save_frame_as_image(frame,frame_index)

            file_path_list.append(file_path)
            
        self.save_file_path_list(file_path_list)
        

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

        file_path = self.save_frame_to_disk(frame,frame_index)

        return file_path

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

        file_name = f"{frame_index:06}.jpg"

        file_path = self.output_dir_path / file_name

        cv2.imwrite(str(file_path), frame)

        return file_path

    def save_file_path_list(self,file_path_list):

        text_path = self.output_dir_path / "images.txt"

        with open(text_path,"w") as f:

            for file_path in file_path_list:

                relative_file_path = file_path.relative_to(self.output_dir_path)
         
                f.write(str(relative_file_path) )
                f.write("\n")
        


if __name__ == "__main__":
    main()