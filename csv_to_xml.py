import csv
from xml.dom import minidom
from xml.dom.minidom import parse
from PIL import Image
import os.path


def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width, height


# print get_num_pixels("/path/to/my/file.jpg")

pat = "/VinBigData/train.csv"
with open(pat, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    # sortedlist = sorted(csv_reader, key=operator.itemgetter(1), reverse=True)
    line_count = 1
    previous = "xxx"
    save_path_file = "yyy"

    for row in csv_reader:
        # print(row["image_id"])
        # print(previous)
        print("*** " + str(line_count) + " ***")
        if row["image_id"] == previous:
            doc = parse(save_path_file)
            annotations = doc.getElementsByTagName('annotation')[0]
            entrys = annotations.getElementsByTagName('object')

            xml2 = root.createElement('object')
            xml3 = root.createElement('name')
            xml3.appendChild(root.createTextNode(row["class_name"]))
            xml2.appendChild(xml3)

            xml3 = root.createElement('pose')
            xml3.appendChild(root.createTextNode('N.A'))
            xml2.appendChild(xml3)
            xml3 = root.createElement('truncated')
            xml3.appendChild(root.createTextNode('N.A'))
            xml2.appendChild(xml3)
            xml3 = root.createElement('difficult')
            xml3.appendChild(root.createTextNode('N.A'))
            xml2.appendChild(xml3)
            xml3 = root.createElement('bndbox')
            xml4 = root.createElement('xmin')
            xml4.appendChild(root.createTextNode(row["x_min"] if row["x_min"] != '' else '0'))
            xml3.appendChild(xml4)
            xml4 = root.createElement('ymin')
            xml4.appendChild(root.createTextNode(row["y_min"] if row["y_min"] != '' else '0'))
            xml3.appendChild(xml4)
            xml4 = root.createElement('xmax')
            xml4.appendChild(root.createTextNode(row["x_max"] if row["x_max"] != '' else '0'))
            xml3.appendChild(xml4)
            xml4 = root.createElement('ymax')
            xml4.appendChild(root.createTextNode(row["y_max"] if row["y_max"] != '' else '0'))
            xml3.appendChild(xml4)
            xml2.appendChild(xml3)

            annotations.insertBefore(xml2, entrys[0])
            doc = doc.childNodes[0].toprettyxml()
            doc = "".join([s for s in doc.splitlines(True) if s.strip("\r\n\t")])
            textfile = open(save_path_file, 'w')
            textfile.write(doc)
            textfile.close()
            continue

        root = minidom.Document()
        dec = root.toxml()
        xml1 = root.createElement('annotation')
        root.appendChild(xml1)

        xml2 = root.createElement('folder')
        xml2.appendChild(root.createTextNode('VinBigData'))
        xml1.appendChild(xml2)

        xml2 = root.createElement('filename')
        xml2.appendChild(root.createTextNode(row["image_id"] + ".dicom"))
        xml1.appendChild(xml2)

        xml2 = root.createElement('path')
        xml2.appendChild(root.createTextNode("don't care"))
        xml1.appendChild(xml2)

        xml2 = root.createElement('source')
        xml3 = root.createElement('database')
        xml3.appendChild(root.createTextNode('Unknown'))
        xml2.appendChild(xml3)
        xml1.appendChild(xml2)

        width = ''
        height = ''
        path_to_file = "/train_jpg/" + str(row['image_id']) + ".jpg"
        # print(str({row['image_id']}) + ".jpg")
        if os.path.exists(path_to_file):
            width, height = get_num_pixels(path_to_file)
            print(width, height)
        else:
            width = 'temp'
            height = 'temp'
        # if os.path.exists(path_to_file) and row["class_name"] != 'No finding':
            # os.remove(path_to_file)
            # print('Bad file found!')
        xml2 = root.createElement('size')
        xml3 = root.createElement('width')
        xml3.appendChild(root.createTextNode(str(width)))
        xml2.appendChild(xml3)
        xml3 = root.createElement('height')
        xml3.appendChild(root.createTextNode(str(height)))
        xml2.appendChild(xml3)
        xml3 = root.createElement('depth')
        xml3.appendChild(root.createTextNode('1'))
        xml2.appendChild(xml3)
        xml1.appendChild(xml2)

        xml2 = root.createElement('segmented')
        xml2.appendChild(root.createTextNode('0'))
        xml1.appendChild(xml2)

        xml2 = root.createElement('object')
        xml3 = root.createElement('name')
        xml3.appendChild(root.createTextNode(row["class_name"]))
        xml2.appendChild(xml3)
        xml3 = root.createElement('pose')
        xml3.appendChild(root.createTextNode('Unspecified'))
        xml2.appendChild(xml3)
        xml3 = root.createElement('truncated')
        xml3.appendChild(root.createTextNode('0'))
        xml2.appendChild(xml3)
        xml3 = root.createElement('difficult')
        xml3.appendChild(root.createTextNode('0'))
        xml2.appendChild(xml3)
        xml3 = root.createElement('bndbox')
        xml4 = root.createElement('xmin')

        xml4.appendChild(root.createTextNode(row["x_min"] if row["x_min"] != '' else '0'))
        xml3.appendChild(xml4)
        xml4 = root.createElement('ymin')
        xml4.appendChild(root.createTextNode(row["y_min"] if row["y_min"] != '' else '0'))
        xml3.appendChild(xml4)
        xml4 = root.createElement('xmax')
        xml4.appendChild(root.createTextNode(row["x_max"] if row["x_max"] != '' else '0'))
        xml3.appendChild(xml4)
        xml4 = root.createElement('ymax')
        xml4.appendChild(root.createTextNode(row["y_max"] if row["y_max"] != '' else '0'))
        xml3.appendChild(xml4)
        xml2.appendChild(xml3)
        xml1.appendChild(xml2)

        xml_str = root.toprettyxml(indent="\t")[len(dec) + 1:]

        save_path_file = "/train_xml/" + row["image_id"] + ".xml"

        with open(save_path_file, "w") as f:
            f.write(xml_str)

        previous = row["image_id"]
        line_count += 1
