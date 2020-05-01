import os

# 指定目录位置
path = 'D:\mylearn\\NCF_Pytorch_Neural-Collaborative-Filtering\pretrain'
# 遍历所有文件
file_names = os.listdir(path)
file_list = []
# 迭代每个文件名
print(len(file_names))
file_1 = file_names[0]
file_1_split = file_1.split('_')
for i, item in enumerate(file_names, 1):
    file_2 = item
    file_2_split = file_2.split('_')
    if file_1_split[0:2] != file_2_split[0:2]:
        file_list.append(file_1)
    file_1 = file_2
    file_1_split = file_2_split
file_list.append(file_1)
for e in file_list:
    file_names.remove(e)
print(len(file_names))
for file in file_names:
    os.remove(path+'\\'+file)
print('delete')
