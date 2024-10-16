import os
import os.path


def dfs_showdir(path, depth):
    if depth == 0:
        print("root:[" + path + "]")
    # print("当前文件路径是{}，包含文件有{}。".format(path, os.listdir(path)))

    for item in os.listdir(path):
        if item in ['logs','.git', '.gitignore', '.idea', '__pycache__', 'checkpoints', 'dataset', 'docs', 'venv']:
            continue

        print("|   " * depth + "+--" + item)

        new_item = path + '/' + item
        if os.path.isdir(new_item):
            dfs_showdir(new_item, depth + 1)


if __name__ == '__main__':
    dfs_showdir('../src', 0)

# def list_directories(path):
#     for root, dirs, files in os.walk(path):
#
#         # 输出当前目录的文件夹名称
#         for dir in dirs:
#             print(os.path.join(root, dir))
#
# if __name__ == '__main__':
#     # 替换为你的项目路径
#     project_path = '../src'
#     list_directories(project_path)
