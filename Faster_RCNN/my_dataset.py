from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
from Pytorch_Detection.Faster_RCNN import transforms
from Pytorch_Detection.Faster_RCNN.draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random


class VOC2012DataSet(Dataset):
    """读取解析VOC2012数据集"""

    def __init__(self, voc_root, transforms, txt_name: str = "train.txt"):
        # -----------------------------------------------------------------------#
        # 设定文件路径
        # -----------------------------------------------------------------------#
        self.root = os.path.join(voc_root, "../../DataSet/Voc_Dataset/VOCdevkit", "VOC2012")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # -----------------------------------------------------------------------#
        # 读取train.txt 和 val.txt. txt_name = train.txt
        # -----------------------------------------------------------------------#
        txt_path = os.path.join(self.root, "ImageSets", "Main", "train.txt")
        assert os.path.exists(txt_path), "not found {} file".format(txt_name)

        # -----------------------------------------------------------------------#
        # 读取xml文件,将其中的换行符通过line.strip()去掉,再加上.xml后缀,得到xml_list
        # -----------------------------------------------------------------------#
        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # -----------------------------------------------------------------------#
        # 检查文件中是否有信息存在
        # -----------------------------------------------------------------------#
        assert len(self.xml_list) > 0, "in {} file does not find any information".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file".format(xml_path)

        # -----------------------------------------------------------------------#
        # 读取类别字典的json文件,得到class_dict
        # -----------------------------------------------------------------------#
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, item):
        # 获取xml文件路径,打开xml文件,
        xml_path = self.xml_list[item]
        with open(xml_path) as fid:
            xml_str = fid.read()
        # 通过etree读取文件
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([item])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    # 解析xml文件,以字典形式存储
    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}


# read class_indict
category_index = {}
try:
    json_file = open('./pascal_voc_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

# load train data set
train_data_set = VOC2012DataSet(os.getcwd(), data_transform["train"], True)
print(train_data_set)
print(len(train_data_set))
for index in random.sample(range(0, len(train_data_set)), k=5):
    img, target = train_data_set[index]
    img = ts.ToPILImage()(img)
    draw_box(img,
             target["boxes"].numpy(),
             target["labels"].numpy(),
             [1 for i in range(len(target["labels"].numpy()))],
             category_index,
             thresh=0.5,
             line_thickness=5)
    plt.imshow(img)
    plt.show()
