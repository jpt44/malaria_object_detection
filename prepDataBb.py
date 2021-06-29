import json
import os
from multiprocessing import Pool
import tensorflow as tf
from PIL import Image
from absl import flags
from dicttoxml import dicttoxml
from createTFRecord import create_tf_example, split, xml_to_csv

parDir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))  # parent directory

flags.DEFINE_string('set_type', None, 'Type "test" to create the test set or "training" to create the training set')

FLAGS = flags.FLAGS


def imgHeightWidth(path: str):
    with Image.open(path) as im:
        w, h = im.size

    return w, h


def writePbtxt(pth: str, l: list):
    s = ""
    for i in range(len(l)):
        s = s + "item {\n"
        s = s + f"{' ' * 4}name: '{l[i]}'\n"
        s = s + f"{' ' * 4}id: {i + 1}\n"
        s = s + "}\n"

    with open(pth, 'w') as f:
        f.write(s)

    return


def dictToXml(d: dict):
    """
    Converts python dictionary to XML format for TF2 Object Detection API
    :param d: {
               folder: <folder name where image is located>
               filename: <image name>,
               path: <path to image with image name>,
               source: {database: Unknown},
               width: width of image in pixels (horizontal)
               height: height of image in pixels (vertical)
               depth: number of channels (3 for color images)
               segmented: 0,
               objects: list of dictionaries [{name: <name of object>, bndbox: {xmin:<xmin>, ymin:<ymin>,
               xmax:<xmax>, ymax:<ymax>}}, {name: <>, bndbox: {}},]
              }
    :return: Nothing
    """

    encoding = "UTF-8"

    xml = dicttoxml(d, custom_root='annotation', attr_type=False, item_func=lambda x: None)

    # dicttoxml prints "item" for lists, the following code removes that====
    xml = xml.decode(encoding)
    xml = xml.replace('<object>', '').replace('</object>', '')
    xml = xml.replace('<None>', '<object>').replace('</None>', '</object>')
    xml = xml.encode(encoding)
    # dicttoxml prints "item" for lists, the above code removes that====

    with open(os.path.join("malaria", d["folder"], d["filename"].replace(".png", ".xml")), "wb") as f:
        f.write(xml)


def createXml(dataJsonPth, imagesPth, set_type, types, start, stride):
    dataJson = json.load(open(dataJsonPth, "r"))

    imgSet = set(os.listdir(imagesPth))

    # iterate through each image
    ct = 0
    for d in range(start, len(dataJson), stride):
        imgPath = f'malaria{dataJson[d]["image"]["pathname"]}'
        imgName = imgPath.split("/")[-1]

        if imgPath.endswith(".jpg"):
            continue

        boxes = dataJson[d]["objects"]

        if imgName in imgSet:
            ct += 1
            YDict = {
                "folder": set_type,
                "filename": imgName,
                "path": f"{parDir}/malaria/{set_type}/{imgName}",
                "source": {"database": "Unknown"},
                "size": {"width": imgHeightWidth(f"malaria/{set_type}/{imgName}")[0],
                         "height": imgHeightWidth(f"malaria/{set_type}/{imgName}")[1],
                         "depth": 3},
                "segmented": 0,
                "object": []
            }
        else:
            continue

        # iterate through each bounding box in each image

        if len(boxes) < 1:
            print(f"{imgName} doesn't have bounding boxes")
            continue

        for b in range(0, len(boxes)):

            topLeft = boxes[b]["bounding_box"]["minimum"]
            bottomRight = boxes[b]["bounding_box"]["maximum"]  # c is along width, r is along height
            celltype = boxes[b]["category"]

            if celltype not in types:
                continue

            xmin, ymin = topLeft["c"], topLeft["r"]  # x along horizontal axis, i.e. width
            xmax, ymax = bottomRight["c"], bottomRight["r"]

            tempDict = {"name": celltype,
                        "pose": "Unspecified",
                        "truncated": 0,
                        "difficult": 0,
                        "bndbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}}

            YDict["object"].append(tempDict)

        dictToXml(YDict)
    return ct


def main(unused_argv):
    flags.mark_flag_as_required('set_type')
    tf.config.set_soft_device_placement(True)

    # we will exclude "difficult" and "unknown"
    types = {"schizont": 2, "gametocyte": 3, "ring": 4, "trophozoite": 5, "red blood cell": 0, "leukocyte": 1}

    # write the label map file
    j = [""] * len(types.keys())
    for k, v in types.items():
        j[v] = k

    os.makedirs("malaria/annotations", exist_ok=True)
    writePbtxt("malaria/annotations/label_map.pbtxt", j)

    # Create XML files
    workers = min(6, os.cpu_count())
    funcArgs = [(f"malaria/training.json", f"malaria/{FLAGS.set_type}", FLAGS.set_type, types, i, workers) for i in
                range(workers)]

    with Pool(workers) as pool:
        res = pool.starmap(createXml, funcArgs)

    print(f"Created XML files for {sum(res)} {FLAGS.set_type} images in malaria/{FLAGS.set_type}")

    # Create TF Record Files
    output_path = os.path.join(parDir, "malaria", "annotations", f"{FLAGS.set_type}.record")
    path = os.path.join(parDir, "malaria", FLAGS.set_type)
    label_path = os.path.join(parDir, "malaria", "annotations", "label_map.pbtxt")

    writer = tf.io.TFRecordWriter(output_path)
    examples = xml_to_csv(path)
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, path, label_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f'Successfully created the TFRecord file: {output_path}')


if __name__ == "__main__":
    tf.compat.v1.app.run()
