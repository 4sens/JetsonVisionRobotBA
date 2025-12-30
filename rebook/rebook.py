from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data=r"C:\Users\noovelUser\Documents\YOLO\rebook\dataset\data.yaml",
    epochs=600,
    imgsz=600,
    patience=100,
    batch=8,
    name="reebok_shoe_v5",       
    project=r"C:\Users\noovelUser\Documents\YOLO\rebook\runs",  


    augment=True,
    fliplr=0.5,     # horizontal flip probability
    flipud=0.0,     # vertical flips (für Schuhe meist aus lassen)
    degrees=30.0,   # rotation range (-10° bis +10°)
    scale=0.50,     # scale jitter
    shear=0.0,
    perspective=0.0,

    # color jitter
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)