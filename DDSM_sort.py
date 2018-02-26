import os
import shutil
import stat
import get_file as gf


#changes ddsm variable to not have anything with cc in it
#quick fix
def weed_out(DDSM):
    DDSM = [x for x in DDSM if not "CC" in x]


#use if storage is running out - deletes the cc folders
def weed_out_del(DDSM):
    startStr = "C:/Users/adven/Google Drive/9th Grade Files/srp rip us/datasets/CBIS-DDSM/Mass_Training_Full/CBIS-DDSM"
    DDSM = os.listdir(startStr)
    while len([s for s in DDSM if "CC" in s])!= 0:
        for i in DDSM:
            if "CC" in i:
                print(startStr+"/"+i)
                pathf = gf.get_file(startStr+"/"+i)
                os.chmod(startStr+"/"+i, stat.S_IWUSR)
                os.chmod(pathf, stat.S_IWUSR)
                if os.access(pathf + "/000000.dcm", os.F_OK):
                    os.chmod(pathf + "/000000.dcm", stat.S_IWUSR)
                """
                print os.access(pathf, os.W_OK)
                if os.access(pathf+"/000000.dcm", os.F_OK):
                    os.remove(pathf+"/000000.dcm")
                os.removedirs(pathf)
                """
                shutil.rmtree(startStr+"/"+i, ignore_errors=True)
        DDSM = os.listdir(startStr)
