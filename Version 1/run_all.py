import my_video_capture
import main_file
import play_my_video

output_folder = "D:/College Studies/4th Year/7th Sem/Major Project"  # Replace 'path_to_your_folder' with your desired folder path
video_duration_sec = 5  # Change the duration of the video in seconds

# Construct the full output path
output_path = f"{output_folder}/input_video.mp4"
if __name__ == "__main__":
    my_video_capture.record_video(output_path, video_duration_sec)
    main_file.execute()
    play_my_video.play()
