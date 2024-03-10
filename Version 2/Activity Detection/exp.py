from ultralytics import YOLO

model = YOLO("best2.pt")

results = model(
    source="D:\\College Studies\\4th Year\\7th Sem\\Major Project\\new_vid.mp4",
    show=True,
    conf=0.25,
    save=True,
)
