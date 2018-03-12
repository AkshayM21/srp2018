import pydicom
import get_file


def get_roi_cropped(init_folder, DDSM_main):
    DDSM = get_file.get_full_path_folder(init_folder)
    DDSM_ROI = []
    for i in range(len(DDSM))[::-1]:
        ds = pydicom.dcmread(DDSM[i]+"000000.dcm")
        try:
            if is_mask_cropped(ds.pixel_array.copy()):
                DDSM_ROI.append(DDSM[i]+"000000.dcm")
            else:
                try:
                    x = pydicom.dcmread(DDSM[i]+"000001.dcm")
                    DDSM_ROI.append(DDSM[i]+"000001.dcm")
                except IOError:
                    DDSM_ROI.append(DDSM[i]+"000000.dcm")
                    print("went here")
        except AttributeError:
            try:
                DDSM_main.remove(getDDSMequivalent(DDSM[i]))
            except ValueError:
                print(DDSM[i])
                print(getDDSMequivalent(DDSM[i]))
            #print("uh oh at "+i)
    for j in range(len(DDSM_ROI))[::-1]:
        try:
            try:
                x = pydicom.dcmread(DDSM_ROI[j]).pixel_array
            except AttributeError:
                try:
                    DDSM_main.remove(getDDSMequivalent(DDSM_ROI[j]))
                except ValueError:
                    blahblah = 1
                DDSM_ROI.remove(DDSM_ROI[j])
        except IndexError:
            blahblah = 1
    print(len(DDSM_ROI))
    print(len(DDSM_main))
    return DDSM_ROI, DDSM_main


def get_roi(init_folder):
    DDSM = get_file.get_full_path_folder(init_folder)
    DDSM_ROI = []
    print(DDSM)
    for i in DDSM[::-1]:
        ds = pydicom.dcmread(i+"000000.dcm")
        try:
            if is_mask(ds.pixel_array.copy()):
                DDSM_ROI.append(i+"000000.dcm")
            else:
                try:
                    x = pydicom.dcmread(i+"000001.dcm")
                    DDSM_ROI.append(i+"000001.dcm")
                except IOError:
                    DDSM_ROI.append(i+"000000.dcm")
                    print("went here for "+i)
        except AttributeError:
            blahblah = 1
            print("uh oh at "+i)
    for j in DDSM_ROI[::-1]:
        try:
            x = pydicom.dcmread(j).pixel_array
        except AttributeError:
            DDSM_ROI.remove(j)
            print("uh oh2 at "+j)
    print(len(DDSM_ROI))
    return DDSM_ROI


def get_roi_single(init_folder):
    folder = get_file.get_folder(init_folder)
    ds = pydicom.dcmread(folder+"000000.dcm")
    if is_mask(ds.pixel_array):
        return folder+"000000.dcm"
    else:
        return folder+"000001.dcm"


def is_mask_cropped(pixel_array):
    return pixel_array.shape[0] == 224 and pixel_array.shape[1] == 224


def is_mask(pixel_array):
    return pixel_array.shape[0] > 1000 and pixel_array.shape[1] > 1000


def make_mask(pixel_array):
    y = 0
    x = 0
    for pixels in range(pixel_array.shape[0]*pixel_array.shape[1]):
        if pixel_array[y, x] != 0:
            pixel_array[y,x] = 1
        if x == pixel_array.shape[0]-1:
            y += 1
            x = 0
            continue
        x+=1
    return pixel_array


def getDDSMequivalent(filename):
    import get_file
    list = filename.split("/")
    return get_file.get_file("C:/Srp 2018/Training-Full/"+list[4][0:len(list[4])-2])



