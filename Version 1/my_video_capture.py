import cv2


def record_video(output_path, duration_sec):
    # Open the default camera (usually the laptop's camera)
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    start_time = cv2.getTickCount() / cv2.getTickFrequency()

    while True:
        ret, frame = cap.read()

        # Write the frame to the video
        out.write(frame)

        # Check if the duration is reached
        elapsed_time = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
        if elapsed_time >= duration_sec:
            break

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release everything when finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Change these values as needed
output_folder = "D:/College Studies/4th Year/7th Sem/Major Project"  # Replace 'path_to_your_folder' with your desired folder path
video_duration_sec = 5  # Change the duration of the video in seconds

# Construct the full output path
output_path = f"{output_folder}/input_video.mp4"

# Record the video for the specified duration
record_video(output_path, video_duration_sec)
