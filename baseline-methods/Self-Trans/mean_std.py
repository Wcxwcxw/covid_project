import normalize
import torch
import torch.utils.data.dataloader as dataloader

#
# from torchvision import transforms as T, transforms
# from torchvision.datasets import ImageFolder
#
# transform = T.Compose([T.Resize(256), T.ToTensor()])
# train_transformer = transforms.Compose([
#     transforms.Resize(256),
#     transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     # transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor()
# ])
# train_dataset = ImageFolder(root="/home/wangchenxu/covid/Images-processed", transform=transform)
# # train_dataset = ImageFolder(root="LUNA", transform=transform)
# train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
#
#
# def get_mean_std(loader):
#     # VAR[X]=E[X**2]-E(X)**2
#     # 公式推导参考https://zhuanlan.zhihu.com/p/35435231
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#     for data, _ in loader:
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#         num_batches += 1
#     mean = channels_sum / num_batches
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
#     return num_batches, mean, std
#
#
# num, mean, std = get_mean_std(train_loader)
# print(num)
# print(mean)
# print(std)

t = torch.tensor([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]])
p = torch.flatten(t)
print(p)