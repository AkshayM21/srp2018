import os

def get_file(folderpath):
    if os.access(folderpath, os.F_OK):
        for x in range(0, 3):
            folderpath += "/" + os.listdir(folderpath)[0];
    return folderpath




