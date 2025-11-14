import splitfolders

splitfolders.ratio(
    "data\garbage_classification_v3",
    output="dataset_yolo_v3",
    seed=42,
    ratio=(0.7, 0.2, 0.1)
)