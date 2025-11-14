import splitfolders

splitfolders.ratio(
    "data\garbage-dataset",
    output="dataset_yolo_v2",
    seed=42,
    ratio=(0.7, 0.2, 0.1)
)