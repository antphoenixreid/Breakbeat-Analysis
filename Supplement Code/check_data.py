import os

def main():
    exts = {'.txt'}
    for directory in get_directories_without_exts('E:\Engineering\Signal Processing\Personal Projects\Breakbeat Analysis\Data', exts):
        print(directory)

def get_directories_without_exts(root, exts):
    for root, dirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1] in exts:
                break
        else:
            yield root

if __name__ == '__main__':
    main()