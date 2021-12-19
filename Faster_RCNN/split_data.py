import os
import random
import cv2


def main():
    random.seed(0)
    files_path = "../../DataSet/Voc_Dataset/VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path:'{}' does not exist".format(files_path)
    val_rate = 0.5
    # 用"."做分割,将文件名分为两部分.[0]表示名称,[1]表示后缀
    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    # 从全部的文件[0,file_num]进行采样,采样率为val_rate
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []

    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
