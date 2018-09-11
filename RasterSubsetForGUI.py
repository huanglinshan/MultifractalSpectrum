# Script to extract a spatial subset from a raster.
import os
from osgeo import gdal
from osgeo import ogr
import platform
# Get all the .tif files in the folder
# def file_name(file_dir, type):
#     """Returns the lists of a specific type of files in the folder.

#     file_dir  -  folder of dataset
#     type - the file type (str)
#     """
#     L=[] 
#     for root, dirs, files in os.walk(file_dir):  
#         for file in files:  
#             if os.path.splitext(file)[1] == type:  
#                     L.append(os.path.join(root, file))  
#     return L

# Extract subsets in the BBox.
def extract_subset_from_folder(in_ds_list, out_folder, BBox):
    """Extract subset of dataset in a file folder and put it in another folder.

    in_folder -  original folder
    out_folder - output folder
    BBox - the binding box
    """

    # Loop through the tif files in the directory.
    for i in range(len(in_ds_list)):
        in_ds = gdal.Open(in_ds_list[i])
        print('Extracting ' + in_ds_list[i] + '...')
        
        # Geotransform the original map
        in_gt = in_ds.GetGeoTransform()
        # Create an inverse geotransform for the raster. This converts real-world coordinates to pixel offsets.
        inv_gt = gdal.InvGeoTransform(in_gt)
        if gdal.VersionInfo()[0] == '1':
            if inv_gt[0] == 1:
                inv_gt = inv_gt[1]
            else:
                raise RuntimeError('Inverse geotransform failed')
        elif inv_gt is None:
            raise RuntimeError('Inverse geotransform failed')
        
        # coordinates of upperleft and lowerright points of binding box
        box_ulx, box_uly, box_lrx, box_lry = BBox[0][0],BBox[0][1],BBox[1][0],BBox[1][1]

        # Get the offsets that correspond to the bounding box corner coordinates.
        offsets_ul = gdal.ApplyGeoTransform(
            inv_gt, box_ulx, box_uly)
        offsets_lr = gdal.ApplyGeoTransform(
            inv_gt, box_lrx, box_lry)

        # The offsets are returned as floating point, but we need integers.
        off_ulx, off_uly = map(int, offsets_ul)
        off_lrx, off_lry = map(int, offsets_lr)

        # Compute the numbers of rows and columns to extract, based on the offsets.
        rows = off_lry - off_uly
        columns = off_lrx - off_ulx

        # Create an output raster with the correct number of rows and columns.
        gtiff_driver = gdal.GetDriverByName('GTiff')
        
        # Check the platform and define in_fn
        sysstr = platform.system()
        if(sysstr =="Windows"):
            in_fn = in_ds_list[i].split('\\')[-1]
            print ("The platform is Windows. Please use '\\' to split the path.")
        elif(sysstr == "Linux"):
            in_fn = in_ds_list[i].split('/')[-1]
            print ("The platform is Linus. Please use '/' to split the path.")
        else: # Other platform might get wrong!
            print ("Other System tasks, might be wrong!")

        out_fn_ori = in_fn.split('.')[0]
        out_fn = out_fn_ori + 'InBox.tif'
        out_ds = gtiff_driver.Create(os.path.join(out_folder,out_fn), columns, rows, 1)
        out_ds.SetProjection(in_ds.GetProjection())

        # Convert the offsets to real-world coordinates for the georeferencing info.
        # We can't use the coordinates above because they don't correspond to the
        # pixel edges.
        subset_ulx, subset_uly = gdal.ApplyGeoTransform(
            in_gt, off_ulx, off_uly)
        out_gt = list(in_gt)
        out_gt[0] = subset_ulx
        out_gt[3] = subset_uly
        out_ds.SetGeoTransform(out_gt)

        # Loop through the only band.
        for i in range(1, 2):
            in_band = in_ds.GetRasterBand(i)
            out_band = out_ds.GetRasterBand(i)

            # Read the data from the input raster starting at the computed offsets.
            data = in_band.ReadAsArray(
                off_ulx, off_uly, columns, rows)
            # Multiply 
            # data = data*255
            # Write the data to the output, but no offsets are needed because we're
            # filling the entire image.
            out_band.WriteArray(data)

        del out_ds

# # =============================================Main=======================================
# # Coordinates for the bounding box to extract.


# # Change the directory.
# in_folder = '/home/linbq/Documents/TestingFromW1/00BeijingLanduse/RawData/TiffMap1984-2015'
# out_folder = '/home/linbq/Documents/TestingFromW1/00BeijingLanduse/RawData/TiffMap1984-2015_InBox'

# # Extract the subset to the out_folder.
# extract_subset_from_folder(in_folder, out_folder, BBox)






# def ExtractSubset(dir,Box):

# in_ds = gdal.Open('BJ1984ReProj30WithinAdmin1.tif')


