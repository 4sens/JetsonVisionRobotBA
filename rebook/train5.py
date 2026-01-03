import torch

# YOLOv5 Model laden
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Training starten
results = model.train(
    data=r".\dataset\data.yaml",
    epochs=600,
    imgsz=600,
    patience=100,
    batch_size=8,
    name="reebok_shoe_y5_v1",
    project=r".\runs",
    
    # Augmentation aktivieren
    augment=True,
    
    # Flip Augmentations
    fliplr=0.5,         # horizontal flip probability
    flipud=0.0,         # vertical flip (bei Schuhen meist deaktiviert)
    
    # Geometrische Augmentations
    degrees=30.0,       # rotation range (+/- degrees)
    translate=0.1,      # translation (+/- fraction)
    scale=0.5,          # scale jitter (+/- gain)
    shear=0.0,          # shear (+/- degrees)
    perspective=0.0,    # perspective warp (+/- fraction)
    
    # Color Augmentations (HSV)
    hsv_h=0.015,        # hue jitter
    hsv_s=0.7,          # saturation jitter
    hsv_v=0.4,          # value (brightness) jitter
    
    # Weitere wichtige Parameter
    mosaic=1.0,         # mosaic augmentation probability
    mixup=0.0,          # mixup augmentation probability
)