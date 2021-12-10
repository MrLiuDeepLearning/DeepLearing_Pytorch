import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
import os
from Pytorch_ResNet_V3.model_50 import ResNet50
from tqdm import tqdm


def main():
    batchsz = 128

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    image_path = data_root + "/data_CIFAR10"
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist".format(image_path)
    train_dataset = datasets.CIFAR10(root=image_path, train=True, transform=transforms.Compose(
                                    [transforms.Resize((32, 32)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                                     download=True)
    train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)

    test_dataset = datasets.CIFAR10(root=image_path, train=False, transform=transforms.Compose(
                                 [transforms.Resize((32, 32)), transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                                  download=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)

    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = ResNet50().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    epochs = 5

    for epoch in range(epochs):

        model.train()
        train_bar = tqdm(train_loader)
        for batchidx, (x, label) in enumerate(train_bar):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        print(epoch + 1, 'train loss:', loss)

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            test_bar = tqdm(test_loader)
            for x, label in test_bar:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)  # print(correct)
                test_bar.desc = "val   epoch[{}/{}] acc:{:.3f}".format(epoch + 1, epochs, total_correct / total_num)

            acc = total_correct / total_num
            print(epoch+1, 'test acc:', acc)


if __name__ == '__main__':
    main()
