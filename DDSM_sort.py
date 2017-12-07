import os
import shutil

#changes ddsm variable to not have anything with cc in it
#quick fix
def weed_out(DDSM):
    DDSM = [x for x in DDSM if not "CC" in x]

#use if storage is running out - deletes the cc folders
def weed_out_del(DDSM):
    for i in DDSM:
        if "CC" in i:
            shutil.rmtree(i, ignore_errors=true)

