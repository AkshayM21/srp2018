import os

def get_file(folderpath):
    if os.access(folderpath, os.F_OK):
        for x in range(0, 2):
            if not len(os.listdir(folderpath)) == 0:
                folderpath += "/" + os.listdir(folderpath)[0]
                print folderpath
            else:
                break
    return folderpath


def get_full_path(init_folder):
    DDSM = []
    if os.access(init_folder, os.F_OK):
        folderList = os.listdir(init_folder)
        for i in folderList:
            DDSM.append(get_file(i))
    return DDSM
