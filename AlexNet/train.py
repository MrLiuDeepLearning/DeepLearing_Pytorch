import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from Pytorch_AlexNet_V1.model import AlexNet

import os, json, time


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
                                     transforms.RandomHorizontalFlip(),
                                     # **功能：**依据概率p对PIL图片进行水平翻转，p默认0.5
                                     transforms.ToTensor(),
                                     # 1. 是将输入的数据shape W，H，C ——> C，W，H. 2. 将所有数除以255，将数据归一化到【0，1】
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     # 功能：逐channel的对图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
                                     ]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = data_root + "/data_flower"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 训练数据集的设定与读取
    train_dataset = datasets.ImageFolder(root=image_path + "/train", transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=0
                                               )
    train_num = len(train_dataset)
    print(train_num)

    # 训练数据集的设定与读取
    validate_dataset = datasets.ImageFolder(root=image_path + "/val", transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=0)
    val_num = len(validate_dataset)
    print(val_num)

    # 获取名称所对应的索引
    flower_list = train_dataset.class_to_idx
    # 遍历获得的字典，并且将其反过来，也就是序号与名称调换
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 编码为json格式，
    json_str = json.dumps(cla_dict, indent=4)
    # 保存json到相应的文件当中
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32

    test_data_iter = iter(validate_loader)
    test_image, test_label = test_data_iter.next()

    # def imshow(img):
    #     img = img / 2 + 0.5
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))
    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    pata = list(net.parameters())  # 查看模型的参数
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 优化函数优化的对象是全体参数

    save_path = './AlexNet.pth'
    best_acc = 0.0
    # epochs = 10
    train_steps = len(train_loader)

    for epoch in range(10):
        net.train()
        running_loss = 0.0  # 训练过程中的平均损失
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()  # 清空之前的梯度信息
            outputs = net(images.to(device))  # 正向传播得到了输出output
            loss = loss_function(outputs, labels.to(device))  # 计算损失，同时将labels送进device
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 打印训练进度
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) + 50)
            print("\rtrain loss : {:^3.0f} % [{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print(time.perf_counter() - t1)

        # 训练一次之后就进行验证
        #     net.eval()
        #     acc = 0.0
        #     with torch.no_grad():  # 不进行梯度更新
        #         for data_test in validate_loader:
        #             test_images, test_labels = data_test
        #             outputs = net(test_images.to(device))
        #             predict_y = torch.max(outputs, dim=1)[1]  # torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index（也就是predicted）
        #             acc += (predict_y == test_labels.to(device).sum().item())  # acc是验证正确的样本个数
        #         accurate_test = acc / val_num
        #         if accurate_test > best_acc:
        #             best_acc = accurate_test
        #             torch.save(net.state_dict(), save_path)
        #         print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #               (epoch + 1, running_loss / step, acc / val_num))
        #
        # print('Finished Training')
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
