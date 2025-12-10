from ultralytics import YOLO

# Load a model

model_size = "yolo11n.pt"
data_path = "yolo_data\data.yaml" 
imgsz = 420
batch_size = 16
epochs = 100
patience = 100
name_run = "2-a3-a4" 
half_precision = True
save_period = 5
workers=8

model = YOLO(model_size)

# Train the model
results = model.train(
    data=data_path,
    imgsz=imgsz,
    batch=batch_size,
    epochs=epochs,
    name=name_run,
    patience=patience,
    half=half_precision,
    save_period=save_period,
    project="./runs/segmentation",
    exist_ok=True,
    workers=workers,

    verbose=True,
    plots=True,
)