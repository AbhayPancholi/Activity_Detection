import cv2
import os
import tensorflow
import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import backend as k
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tempfile
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# from six.moves.urllib.request import urlopen
from six import BytesIO
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import time

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())


class Activity_Detection:
    def __init__(self):
        self.module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        self.model = hub.load(self.module_handle)
        self.detector = self.model.signatures["default"]
        self.act_model = keras.models.load_model(
            "D:/College Studies/4th Year/7th Sem/Major Project/activity_detecton.h5"
        )

    def video_to_frames(
        self,
        video_path,
        output_path="D:/College Studies/4th Year/7th Sem/Major Project/Video to frames",
        frame_skip=100,
    ):
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0

        while success:
            if count % frame_skip == 0:
                cv2.imwrite(output_path + f"/frame{count}.jpg", image)
            success, image = vidcap.read()
            count += 1

    def display_image(self, image):
        fig = plt.figure(figsize=(20, 15))
        plt.grid(False)
        plt.imshow(image)

    def resize_image(self, input_image, new_width=256, new_height=256, display=False):
        pil_image = Image.open(input_image)
        pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
        _, output_filename = tempfile.mkstemp(suffix=".jpg")
        pil_image.save(output_filename, format="JPEG", quality=90)
        print("Image resized and saved to %s." % output_filename)
        if display:
            self.display_image(pil_image)

        return output_filename

    def draw_bounding_box_on_image(
        self,
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color,
        font,
        thickness=4,
        display_str_list=(),
    ):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size

        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )

        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
            width=thickness,
            fill=color,
        )

        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = top + total_display_str_height

        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [
                    (left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom),
                ],
                fill=color,
            )
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font,
            )
            text_bottom -= text_height - 2 * margin

    def draw_boxes(
        self, image, boxes, class_names, scores, max_boxes=10, min_score=0.1
    ):
        colors = list(ImageColor.colormap.values())

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                25,
            )
        except IOError:
            print("Font not found, using default font.")
            font = ImageFont.load_default()

        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                display_str = "{}".format(class_names[i])
                color = colors[hash(class_names[i]) % len(colors)]
                image_pil = Image.fromarray(np.uint8(image)).convert("RGB")

                self.draw_bounding_box_on_image(
                    image_pil,
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                    color,
                    font,
                    display_str_list=[display_str],
                )
                np.copyto(image, np.array(image_pil))

        return image

    def extract_persons_images(self, image, boxes, path):
        image = Image.open(path)
        images_list = []
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            im_width, im_height = image.size
            left, right, top, bottom = (
                int(xmin * im_width),
                int(xmax * im_width),
                int(ymin * im_height),
                int(ymax * im_height),
            )

            roi = image.crop((left, top, right, bottom))

            images_list.append(roi)

        return images_list

    def classify_image(self, img):
        # img = image.load_img(img, target_size=(224, 224))
        # img = tf.image.resize(input_img, (244,224))
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = self.act_model.predict(img_array)

        class_labels = ["Abnormal Activity", "Walking"]
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        return predicted_class_label

    def save_images(self, img, count):
        output_folder = (
            "D:/College Studies/4th Year/7th Sem/Major Project/Frames to video"
        )
        filename = f"output_image {count}.jpg"
        img = Image.fromarray(np.uint8(img))
        output_path = os.path.join(output_folder, filename)
        img.save(output_path)

    def load_img(self, path):
        img = tf.io.read_file(path)

        img = tf.image.decode_jpeg(img, channels=3)

        return img

    def run_detector(self, path, count):
        img = self.load_img(path)

        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

        start_time = time.time()
        result = self.detector(converted_img)
        end_time = time.time()

        result = {key: value.numpy() for key, value in result.items()}

        print("Found %d objects." % len(result["detection_scores"]))
        print("Inference time: ", end_time - start_time)

        arr = np.array(result["detection_class_entities"])
        detection_class_entities = []
        detection_boxes = []
        detection_score = []
        for i in np.where(arr == b"Person")[0]:
            detection_class_entities.append(result["detection_class_entities"][i])
            detection_boxes.append(result["detection_boxes"][i])
            detection_score.append(result["detection_scores"][i])

        detection_class_entities = np.array(detection_class_entities)
        detection_boxes = np.array(detection_boxes)
        detection_score = np.array(detection_score)

        lst_extracted_imgs = self.extract_persons_images(
            img, list(detection_boxes), path
        )

        pred_class_lst = []
        for person_image in lst_extracted_imgs:
            predicted_class = self.classify_image(person_image)
            pred_class_lst.append(predicted_class)
        pred_class_lst = np.array(pred_class_lst)
        # draw predicted boxes over the image
        image_with_boxes = self.draw_boxes(
            img.numpy(), detection_boxes, pred_class_lst, detection_score
        )

        self.save_images(image_with_boxes, count)
        self.display_image(image_with_boxes)

    def detect_img(self, image_url, count):
        start_time = time.time()
        image_path = self.resize_image(image_url, 640, 480)
        self.run_detector(image_path, count)
        end_time = time.time()
        print("Inference time:", end_time - start_time)

    def predict(
        self,
        folder_path="D:/College Studies/4th Year/7th Sem/Major Project/Video to frames",
    ):
        image_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]
        class_lst = []
        count = 0
        for img_file in image_files:
            count += 1
            img_path = os.path.join(folder_path, img_file)
            self.detect_img(img_path, count)

    def delete_all_files_in_folder(self, folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # else:
                #     os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    def images_to_video(
        self,
        images_folder="D:/College Studies/4th Year/7th Sem/Major Project/Frames to video",
        output_video_path="D:/College Studies/4th Year/7th Sem/Major Project/output_video.mp4",
        fps=24,
    ):
        image_files = [
            os.path.join(images_folder, f)
            for f in os.listdir(images_folder)
            if f.endswith(".jpg")
        ]  # Change the extension if needed
        image_files.sort()
        image = cv2.imread(image_files[0])
        height, width, _ = image.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for image_file in image_files:
            image = cv2.imread(image_file)
            out.write(image)
        out.release()


def execute():
    obj = Activity_Detection()
    obj.video_to_frames(
        "D:/College Studies/4th Year/7th Sem/Major Project/Walking video.mp4"
    )
    obj.predict()
    obj.images_to_video()
    obj.delete_all_files_in_folder(
        "D:/College Studies/4th Year/7th Sem/Major Project/Frames to video"
    )
    obj.delete_all_files_in_folder(
        "D:/College Studies/4th Year/7th Sem/Major Project/Video to frames"
    )


execute()
#     obj.delete_all_files_in_folder()
