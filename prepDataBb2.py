import json
import os
from multiprocessing import Pool
import numpy as np
from PIL import Image

parDir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))  # parent directory


def imgHeightWidth(path: str):
    with Image.open(path) as im:
        w, h = im.size

    return w, h


def create_Y_files(dataJsonPth, trainingImagesPath, testImagesPath, types, start, stride):
    dataJson = json.load(open(dataJsonPth, "r"))

    trainingImages = set(os.listdir(trainingImagesPath))
    testImages = set(os.listdir(testImagesPath))

    # iterate through each image
    for d in range(start, len(dataJson), stride):
        imgPath = f'malaria{dataJson[d]["image"]["pathname"]}'
        imgName = imgPath.split("/")[-1]

        if imgPath.endswith(".jpg"):
            continue

        if imgName in trainingImages:
            width, height = imgHeightWidth(f"{trainingImagesPath}/{imgName}")
        elif imgName in testImages:
            width, height = imgHeightWidth(f"{testImagesPath}/{imgName}")
        else:
            print(f"{imgName} not in train or test folders")
            continue

        boxes = dataJson[d]["objects"]

        if len(boxes) < 1:
            print(f"{imgName} doesn't have bounding boxes")
            continue

        bb_arr = np.zeros(shape=(len(boxes), 4), dtype="float32")
        class_arr = np.zeros(shape=(len(boxes), len(types.keys())), dtype="bool")

        # iterate through each bounding box in each image
        for b in range(len(boxes)):

            topLeft = boxes[b]["bounding_box"]["minimum"]  # c is along height, r is along width for PIL
            bottomRight = boxes[b]["bounding_box"]["maximum"]  # c is along width, r is along height for cv2
            celltype = boxes[b]["category"]

            if celltype not in types:
                continue

            class_arr[b, types[celltype]] = 1

            xmin, ymin = topLeft["c"], topLeft["r"]  # x along horizontal axis, i.e. width
            xmax, ymax = bottomRight["c"], bottomRight["r"]  # x along horizontal axis, i.e. width

            # order = class, xmin, ymin, xmax, ymax
            xminNorm, xmaxNorm = xmin / width, xmax / width
            yminNorm, ymaxNorm = ymin / height, ymax / height

            bb_arr[b, :] = [xminNorm, yminNorm, xmaxNorm, ymaxNorm]

        np.save(f"malaria/Y_bb/bb_{imgName.replace('.png', '.npy')}", bb_arr, allow_pickle=True, fix_imports=False)
        np.save(f"malaria/Y_bb/class_{imgName.replace('.png', '.npy')}", class_arr, allow_pickle=True,
                fix_imports=False)


if __name__ == "__main__":
    # we will exclude "difficult" and "unknown"
    types = {"schizont": 2, "gametocyte": 3, "ring": 4, "trophozoite": 5, "red blood cell": 0, "leukocyte": 1}

    os.makedirs("malaria/Y_bb", exist_ok=True)

    # Create Y files
    workers = min(6, os.cpu_count())
    funcArgs = [("malaria/training.json", "malaria/training", "malaria/test", types, i, workers) for i in
                range(workers)]

    with Pool(workers) as pool:
        pool.starmap(create_Y_files, funcArgs)
    
    print("Created bounding box and one-hot class files")
