#-*-coding:utf-8-*-

# Script to extract a spatial subset from a raster.
import os
from osgeo import gdal
import platform

class Clipper():
    #
    def __init__(self, file_name, clip_box, out_folder=os.getcwd()):
        '''
        Initialize.

        :param file_name  - input file name
        :param clip_box   - clip box
                            box_ulx, box_uly, box_lrx, box_lry = clip_box[0][0], clip_box[0][1], clip_box[1][0], clip_box[1][1]
        :param out_folder - output file path
        '''
        self.file_name = file_name
        self.out_file_name = self.file_name.split('.')[0] + '_clip.tif'
        self.clip_box = clip_box
        self.out_folder = out_folder
        self.name = ''
        self.row = 0
        self.column = 0

    def _check_platform(self):
        '''
        Check the platform is "Windows" or "Linux".
        '''
        # Check the platform and define in_fn
        sysstr = platform.system()
        if sysstr == "Windows":
            self.name = self.file_name.split('\\')[-1]
            print("The platform is Windows. Please use '\\' to split the path.")
        elif sysstr == "Linux":
            self.name = self.file_name.split('/')[-1]
            print("The platform is Linux. Please use '/' to split the path.")

    def _get_inv_gt(self):
        """
        Geotransform the original map and create an inverse geotransform for the raster.
        """
        # Geotransform the original map
        self.in_gt = self.in_ds.GetGeoTransform()
        # Create an inverse geotransform for the raster.
        # This converts real-world coordinates to pixel offsets.
        self.inv_gt = gdal.InvGeoTransform(self.in_gt)
        if gdal.VersionInfo()[0] == '1':
            if self.inv_gt[0] == 1:
                self.inv_gt = self.inv_gt[1]
            else:
                raise RuntimeError('Inverse geotransform failed')
        elif self.inv_gt is None:
            raise RuntimeError('Inverse geotransform failed')


    def _get_clip_loc_in_array(self):
        """
        get clip location in array.
        """

        # coordinates of upperleft and lowerright points of binding box
        box_ulx, box_uly, box_lrx, box_lry = self.clip_box[0][0], self.clip_box[0][1], \
                                             self.clip_box[1][0], self.clip_box[1][1]

        # Get the offsets that correspond to the bounding box corner coordinates.
        offsets_ul = gdal.ApplyGeoTransform(self.inv_gt, box_ulx, box_uly)
        offsets_lr = gdal.ApplyGeoTransform(self.inv_gt, box_lrx, box_lry)

        # The offsets are returned as floating point, but we need integers.
        self.off_ulx, self.off_uly = map(int, offsets_ul)
        self.off_lrx, self.off_lry = map(int, offsets_lr)

        # Compute the numbers of rows and columns to extract, based on the offsets.
        self.row = self.off_lry - self.off_uly
        self.column = self.off_lrx - self.off_ulx

    def _read_image(self):
        """
        Read the image
        """
        # Choose the first band.
        in_band = self.in_ds.GetRasterBand(1)
        data = in_band.ReadAsArray(self.off_ulx, self.off_uly, self.column, self.row)
        return data

    def _write_image(self):
        """
        Write the clipped image.
        """
        # Create an output raster with the correct number of rows and columns.
        gtiff_driver = gdal.GetDriverByName('GTiff')
        out_ds = gtiff_driver.Create(os.path.join(self.out_folder, self.out_file_name), self.column, self.row, 1)
        out_ds.SetProjection(self.in_ds.GetProjection())

        # Convert the offsets to real-world coordinates for the georeferencing info.
        # We can't use the coordinates above because they don't correspond to the pixel edges.
        subset_ulx, subset_uly = gdal.ApplyGeoTransform(self.in_gt, self.off_ulx, self.off_uly)
        out_gt = list(self.in_gt)
        out_gt[0] = subset_ulx
        out_gt[3] = subset_uly
        out_ds.SetGeoTransform(out_gt)

        data = self.read_image()
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(data)

        del out_ds

    def _extract_subset_from_file(self):
        '''
        #################### MAIN FUNCTION ###################
                        Clip image within box
        ######################################################
        '''
        self.in_ds = gdal.Open(self.file_name)
        self._check_platform()
        self._get_inv_gt()
        self._get_clip_loc_in_array()
        self._read_image()
        self._write_image()