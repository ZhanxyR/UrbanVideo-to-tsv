import cv2
import base64
import os
import pandas as pd
import warnings
import copy
import io
import csv
import math

from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    return ret

# Main execution block
if __name__ == '__main__':

    # Save pathes
    video_save_path = 'UrbanVideo_bench'
    tsv_path = 'UrbanVideo_bench_test.tsv'
    resize_scale = 0.5  # Resize the image to a smaller size for faster processing.
    frms = 32

    # Dataset path
    folder_path = 'videos'  # Define the folder path where video files are stored.
    QA_df = pd.read_parquet('MCQ.parquet')  # Read the dataset containing questions and metadata from a Parquet file.
    res = copy.deepcopy(QA_df)

    tsv_data = []
    tsv_index = 0

    # Iterate through each question starting from the last valid index.
    for qa_idx in tqdm(range(res.shape[0])):

        # Get the video ID for the current question.
        select_vid_name = res['video_id'].iloc[qa_idx]

        save_images_path = os.path.join(video_save_path, select_vid_name.replace('.mp4', ''))

        # extract frames from video
        if not os.path.exists(save_images_path):
            video = cv2.VideoCapture(os.path.join(folder_path, str(select_vid_name)))
            video_fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second (FPS) of the video.

            frames = []
            while video.isOpened():
                success, frame = video.read()  # Read a frame from the video.
                if not success:
                    break  # Stop reading if there are no more frames.

                frames.append(frame)

            # Release the video file and print the number of frames read.
            video.release()

            # img_nums = len(os.listdir(save_images_path))
            div_num = math.ceil(len(frames) / frms)  # Divide frames into chunks.
            frames_selected = frames[0::div_num]  # Select every nth frame.

            if not os.path.exists(save_images_path):
                os.makedirs(save_images_path)
                
            for i, frame in enumerate(frames_selected):
                save_image_path = os.path.join(save_images_path, f"{i}.jpg")

                if not os.path.exists(save_image_path):
                    cv2.imwrite(save_image_path, frame)

        # Create a prompt for the GPT model to answer questions based on the video.
        prompt = "This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions: \n"
        # prompt += "The template for the answer is: \n\
        #                 Option: []; Reason: []\n\
        #                 where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. Reason explains why you choose this option."
 
        # Add the question from the dataset to the prompt.
        qa = res['question'].iloc[qa_idx]

        if 'Choices:' in qa:
            split_word = 'Choices:'
        elif 'Choice:' in qa:
            split_word = 'Choice:'
        elif 'choose:' in qa:
            split_word = 'choose:'
        elif 'Options:' in qa:
            split_word = 'Options:'
        elif 'Option:' in qa:
            split_word = 'Option:'
        elif 'Choose:' in qa:
            split_word = 'Choose:'

        if qa_idx in [846, 1021, 1216, 1280, 1404, 1419, 1495, 1599, 1922, 2749, 3378, 4449, 4522, 4609, 4679, 4835, 5072, 5079, 5218]:
            # No options
            continue

        question = qa.split(split_word)[0]
        choices = qa.split(split_word)[1].split('\n')
        choices = [c.strip() for c in choices]

        cleaned_choices = []

        def is_upper(c):
            if ord(c) >= 65 and ord(c) <= 90:
                return True
            else:
                return False

        for i in range(len(choices)):
            if len(choices[i]) > 0 and (is_upper(choices[i][0]) or choices[i][1] == '.'):
                cleaned_choices.append(choices[i][2:].strip())

        # Maximum options
        if len(cleaned_choices) < 8:
            for i in range(8 - len(cleaned_choices)):
                cleaned_choices.append('')


        img_nums = len(os.listdir(save_images_path))
        imgs_base64 = []
        imgs_path = []

        for i in range(img_nums):
            save_image_path = os.path.join(save_images_path, f"{i}.jpg")

            img = Image.open(save_image_path)

            # TODO: resize image to a fixed size
            if resize_scale < 1:
                width, height = img.size
                new_width = int(width // (1/resize_scale))
                new_height = int(height // (1/resize_scale))
                img = img.resize((new_width, new_height))

            img_base64 = encode_image_to_base64(img)
            imgs_base64.append(img_base64)
            imgs_path.append(save_image_path)

        answer = res['answer'].iloc[qa_idx]
        prompt += '\n' + question
        category = res['question_category'].iloc[qa_idx]
        options = cleaned_choices

        tsv_data.append([tsv_index, category, imgs_base64, imgs_path, prompt] + options + [answer])
        # tsv_data.append([tsv_index, category, imgs_path, prompt] + options + [answer])

        tsv_index += 1

    with open(tsv_path, 'w', newline='', encoding='utf-8') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        tsv_writer.writerow(['index', 'category', 'image', 'image_path', 'question', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'answer'])
        # tsv_writer.writerow(['index', 'category', 'image_path', 'question', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'answer'])
        tsv_writer.writerows(tsv_data)

    print(f"Saved TSV to {tsv_path}")
