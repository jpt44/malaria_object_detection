import json
import numpy as np

if __name__ == "__main__":

    # we will exclude "difficult" and "unknown"
    types = {"schizont": 2, "gametocyte": 3, "ring": 4, "trophozoite": 5, "red blood cell": 0, "leukocyte": 1}

    dataJson = json.load(open(f"malaria/training.json", "r"))

    # iterate through each image
    for d in range(len(dataJson)):
        imgPath = f'malaria{dataJson[d]["image"]["pathname"]}'
        imgName = imgPath.split("/")[-1][:-4]

        # exclude JPEGs, too faded
        if imgPath.endswith(".jpg"):
            continue

        boxes = dataJson[d]["objects"]

        Yarr = np.zeros(shape=(1, len(types.keys())))

        for b in range(len(boxes)):

            celltype = boxes[b]["category"]

            if celltype in types:
                Yarr[0, types[celltype]] = 1

        # output X for image
        np.save(f"malaria/Y_nobb/{imgName}.npy", Yarr, True, False)  # save as numpy arr
