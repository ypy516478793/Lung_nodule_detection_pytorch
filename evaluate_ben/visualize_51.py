import os
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz


result_dir = "./detector_ben/results/methodist_finetuned_mode3"
bbox_dir = os.path.join(result_dir, "bbox")
filenames = np.load(os.path.join(bbox_dir, "namelist.npy")).tolist()

# Create samples for your data
samples = []
for file in os.listdir(filenames):
    filepath = os.path.join(bbox_dir, file)
    sample = fo.Sample(filepath=filepath)

    # Convert detections to FiftyOne format
    detections = []
    for obj in annotations[filepath]:
        label = obj["label"]

        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        bounding_box = obj["bbox"]

        detections.append(
            fo.Detection(label=label, bounding_box=bounding_box)
        )

    # Store detections in a field name of your choice
    sample["ground_truth"] = fo.Detections(detections=detections)

    samples.append(sample)

# Create dataset
dataset = fo.Dataset("my-detection-dataset")
dataset.add_samples(samples)


# dataset = foz.load_zoo_dataset("quickstart")
# session = fo.launch_app(dataset)

print("")