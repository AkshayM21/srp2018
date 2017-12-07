import os

def get_file(folderpath):
    if os.access(folderpath, os.F_OK):
        for x in range(0, 2):
            if not len(os.listdir(folderpath)) == 0:
                folderpath += "/" + os.listdir(folderpath)[0];
                print folderpath
            else:
                break
    return folderpath




