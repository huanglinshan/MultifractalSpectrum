#-*-coding:utf-8-*-
import platform
import os

class FileList():
    """
        Return the list of files of a specific type in a file folder (e.g.: tif)
    """
    def __init__(self, dir_name, file_type='.tif'):
        self.dir_name = dir_name
        self.file_type = file_type
        self.file_list = []

    def get_file_name(self):
        """
        Return the list of files of a specific type.
        """
        sysstr = platform.system()
        if (sysstr == "Windows"):
            filefoldername = self.dir_name.replace("/", "\\")
            print("The platform is Windows. Please use '\\' to split the path.")
        elif (sysstr == "Linux"):
            print("The platform is Linux. Please use '/' to split the path.")
        else:  # Other platform might get wrong!
            print("Other System tasks, might be wrong!")

        for root, dirs, files in os.walk(self.dir_name):
            for file in files:
                if os.path.splitext(file)[1] == self.file_type:
                    self.file_list.append(os.path.join(root, file))
        return self.file_list