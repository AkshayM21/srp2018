import pydicom

def replace(file1, file2):
    pixel_array = pydicom.dcmread(file1).pixel_array


    ds = pydicom.dcmread(file2)
    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.PixelData = pixel_array.tostring()
    ds.save_as(file2)