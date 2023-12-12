import cv2


# Path to your video file
def play():
    video_path = "D:/College Studies/4th Year/7th Sem\Major Project/output_video.mp4"  # Replace with your video file path

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Play the video frame by frame
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):  # Press 'q' to exit
            break

    # Release the VideoCapture and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


play()
