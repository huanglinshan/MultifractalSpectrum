# -*- coding: utf-8 -*-

from traitsui.api import *
from traits.api import *
from chaco.api import HPlotContainer, ArrayPlotData, Plot
from enable.component_editor import ComponentEditor
from RasterSubsetForGUI import *
from ResampleForGUI import *
from CalculateForGUI import *
from threading import Thread
from time import sleep

import os
import numpy as np
import platform
import winsound

class FileList(HasTraits):
    """
    数据源，data是一个字典，将字符串映射到列表
    file_list是文件夹filefoldername中的所有特定类型（比如.tif文件等）的文件列表
    """
    
    file_list = List(Str)
    type = '.tif'

    def file_name(self, filefoldername, type):
        """返回文件夹中特定类型文件的列表.
        filefoldername  -  folder of dataset
        type - the file type (str)
        """
        sysstr = platform.system()
        if(sysstr =="Windows"):
            filefoldername = filefoldername.replace("/","\\")
            print ("The platform is Windows. Please use '\\' to split the path.")
        elif(sysstr == "Linux"):
            print ("The platform is Linux. Please use '/' to split the path.")
        else: # Other platform might get wrong!
            print ("Other System tasks, might be wrong!")
            

        for root, dirs, files in os.walk(filefoldername):  
            for file in files:  
                if os.path.splitext(file)[1] == type:  
                        self.file_list.append(os.path.join(root, file))  
        return self.file_list  


class DisplayThread(Thread):
    """ Display loop. This is the worker thread that retrieves images
    from the camera, displays them, and spawns the processing job.
    """
    wants_abort = False
    # def process(self, image):
    #     """ Spawns the processing job. """
    #     try:
    #         if self.processing_job.isAlive():
    #             self.show("Processing too slow")
    #             return
    #     except AttributeError:
    #         pass
    #     self.processing_job = Thread(target=process, args=(image,
    #             self.results))
    #     self.processing_job.start()
    def run(self):
        """ Runs the acquisition loop. """
        self.show(self.message_string)
        # while not self.wants_abort:
        #     if self.add_message == "":
        #         continue
        #     else:
        #         self.show(self.add_message)           
        #     sleep(.1)
        # self.show('Calculation stopped')

class Graph(HasTraits):
    """
    绘图组件，包括左边的数据选择控件和右边的绘图控件
    """
    file_list = Instance(FileList) # 原始tif文件所在的文件夹
    #message_box = Instance(MessageBox)

    name = Str(desc=u'输入标签页标题') # 绘图名，显示在标签页标题和绘图标题中

    center_coordinate = Array(dtype=np.float, shape=(1,2)) # 从左至右分别为x,y
    extent = Array(dtype=np.float, shape=(1,2)) # 东西之差（宽度），南北之差(高度)
    clip_filefolder = Directory() # 文件选择 # 切片存放文件夹

    resample_size = Enum("4096","2048","1024","512","256","128","64","32","16")
            # 重采样的尺寸（都是2的倍数）
    resample_filefolder = Directory() # 重采样存放文件夹

    q_range = Array(dtype=np.float, shape=(1,3)) 
            # 从左至右分别为q_min,q_interval,q_max(绝对值不要超过50)
    intercept = Enum("Fix the intercept to zero during OLS",
            "DO NOT fix the intercept to zero during OLS") # 截距是否固定为0
    export_data = Enum("Export the data to .hdf5 file",
            "DO NOT Export the data to .hdf5 file") # 是否输出数据
    export_png = Enum("Export the result to .png file",
            "DO NOT Export the result to .png file") # 是否输出图片结果
    effective_scale_min = Enum("4096","2048","1024","512","256","128","64",
            "32","16","8","4","2","1")
            # OLS的有效尺度（标度区）最小值
    effective_scale_max = Enum("4096","2048","1024","512","256","128","64",
            "32","16","8","4","2","1")
            # OLS的有效尺度（标度区）最大值
             # 记得检验二者的大小
    export_filefolder = Directory() # 存放数据和图片的文件夹

    calculate_button = Button("Start calculation")
    add_message = String()
    message_string = String()
    display_thread = Instance(DisplayThread)

    name_list = List()
    selected_name = Str
    parameters_list = List()
    plot_name = Str()
    calculation_finish = Int
    plot = Instance(HPlotContainer)

    view = View(
        VSplit( # VGroup分为上下两个区域，中间有可调节宽度比例的调节手柄
            # 上边为一个组
            Tabbed(
                Group(
                    Item("name",label=u'输入名称'),   # 绘图名编辑框
                    Item("calculate_button",label=u'点击开始计算'), # 清除按钮
                    Item("message_string",show_label=False,springy=True,style='custom'),
                    label = u"填入名称及开始计算",
                    show_border = True,
                    show_labels = False,
                ),
                Group(
                    Item("center_coordinate",label='The coordinates of the center (x,y)'),
                    Item("extent",label='The width and height of the clip'),
                    Item("clip_filefolder",label=u"选择切片存放文件夹", width=100),
                    label = u"Clip Parameters 切片参数",
                    show_border = True,
                ),
                Group(
                    Item("resample_size",label=u'输出重采样的尺寸（需为2的倍数）'),
                    Item("resample_filefolder",label=u"选择重采样存放文件夹", width=100),
                    label = u"Resample Parameters 重采样参数",
                    show_border = True
                ),
                Group(
                    Heading(u"q_min_____q_interval___q_max"),  # 静态文本
                    Item("q_range",label=u'矩q的范围'),
                    Item("intercept",label=u"是否固定截距"),
                    HGroup(
                        Item("effective_scale_min",label=u"标度区的最小尺寸值"),
                        Item("effective_scale_max",label=u"标度区的最大尺寸值")
                        # 与resample_size不同，这个是参与回归的有效尺寸。都是2的倍数。
                    ),
                    label = u"Calculation Parameters 计算参数",
                    show_border = True
                ),
                Group(
                    Item("export_data",label=u"是否将计算过程数据保存成hdf5文件"),
                    Item("export_png",label=u"是否将计算结果的图片保存为png文件"),
                    Item("export_filefolder",label=u"选择存放数据和图片的文件夹", width=100),
                    label = u"Save Parameters 保存参数",
                    show_border = True
                )
            ),
            # 下边绘图控件
            Group(
                Item('selected_name',editor=EnumEditor(name="object.name_list",
                        format_str = u'%s'),label=u'请选择需要显示谱线的原文件'),
                Heading(u'左侧是全局谱，右侧是局部谱'),
                Item('plot',editor=ComponentEditor(), show_label=False,width=800,
                    height=300, resizable=True)
            ),
            show_border = True, # 显示组的边框
            scrollable = True,  # 组中的控件过多时，采用滚动条
            show_labels = False, # 组中的所有控件都不显示标签  
        ),
        # handler=CalculateBox_Handler
    )
    
    def _plot_default(self): 
        #super(ContainerExample, self).__init__()
        x = np.linspace(-20, 20, 100)
        y = x*0
        plotdata = ArrayPlotData(x=x, y=y)
        scatter = Plot(plotdata)
        scatter.plot(("x", "y"), type="scatter", color="blue")
        # scatter.x_axis()
        # scatter.xlabel(r'$q$')
        # scatter.ylabel('$D(q)$')
        scatter.title = r"$q$ v.s. $D(q)$"

        line = Plot(plotdata)
        line.plot(("x", "y"), type="line", color="blue")
        # line.xlabel(r'$\alpha(q)$')
        # line.ylabel(r'$f(q)$')
        line.title= r'$\alpha(q)$ v.s. $f(q)$'

        container = HPlotContainer(scatter, line)
        #self.plot = container
        return container

    def _calculate_button_fired(self):
        """ Callback of the "start stop acquisition" button. This starts
        the acquisition thread, or kills it.
        """
        self.start_to_calculate()
        #self.message_string = "The calculation is done!\n" \
                #"Please close the window and check out the result.\n"
    def _selected_name_changed(self):
        position = self.name_list.index(self.selected_name)
        self.update(self.parameters_list[position])


    def process(self):
        try:
            if self.processing_job.isAlive():
                self.show("Processing too slow")
            return
        except AttributeError:
            pass
        self.processing_job = Thread(target=self.start_to_calculate)
        self.processing_job.start()


    def add_line(self, string):
        """ Adds a line to the textbox display.
        """
        self.message_string = (string + "\n" + self.message_string)[0:1000]
        

    
    def start_to_calculate(self):
        self.message_string = 'Begin to calculate...'
        sleep(1)

        box_ulx = self.center_coordinate[0][0]-self.extent[0][0]/2
        box_uly = self.center_coordinate[0][1]+self.extent[0][1]/2
        box_lrx = self.center_coordinate[0][0]+self.extent[0][0]/2
        box_lry = self.center_coordinate[0][1]-self.extent[0][1]/2
        # box_ulx, box_uly = 12906625, 4902314
        # box_lrx, box_lry = 13006625, 4802314
        BBox = [(box_ulx, box_uly),(box_lrx, box_lry)]


        self.add_message= "Cliping...\n"
        print(self.add_message)
        extract_subset_from_folder(self.file_list.file_list,self.clip_filefolder,BBox)


        self.add_message= "Resampling...\n"
        print(self.add_message)
        resample_folder(self.clip_filefolder,self.resample_filefolder,
                int(self.resample_size),int(self.resample_size))
        
        # 是Fix intercept 不是截距本身是否为0
        if self.intercept == 'Fix the intercept to zero during OLS':
            self.Fix_intercept = 1
        elif self.intercept == 'DO NOT fix the intercept to zero during OLS':
            self.Fix_intercept = 0
        
        if self.export_data == "Export the data to .hdf5 file":
            self.Save_data = 1
        elif self.export_data == "DO NOT Export the data to .hdf5 file":
            self.Save_data = 0
        
        if self.export_png == "Export the result to .png file":
            self.Save_pic = 1
        elif self.export_png == "DO NOT Export the result to .png file":
            self.Save_pic = 0


        self.add_message= "Calculating...\n"
        print(self.add_message)
        (self.name_list,self.parameters_list) = calculation_and_save(
                self.resample_filefolder,self.export_filefolder,
                int(self.resample_size),int(self.resample_size),
                self.q_range[0],fix_intercept=self.Fix_intercept,
                effective_scale_min=int(self.effective_scale_min),
                effective_scale_max=int(self.effective_scale_max),
                save_data=self.Save_data,save_pic=self.Save_pic
                )


        self.add_message = "The calculation is done!\n" \
                "Please close the window and check out the result.\n"
        print(self.add_message)

        self.calculation_finish = True
        self.message_string = "The calculation is done!\n" \
                "Please close the window and check out the result.\n"
        # 播放结束音乐
        winsound.PlaySound("Bongos.wav",winsound.SND_ASYNC)


    def _calculation_finish_changed(self):
        self.update(self.parameters_list[-1])

    def update(self,parameters):
        """
        重新绘制所有的曲线
        """    
         #super(ContainerExample, self).__init__()
        x = parameters[0]
        y = zip(*parameters[2])[2]
        plotdata1 = ArrayPlotData(x=x, y=y)
        scatter = Plot(plotdata1)
        scatter.plot(("x", "y"), type="scatter", color="blue")

        x = zip(*parameters[3])[2]
        y = zip(*parameters[4])[2]
        plotdata2 = ArrayPlotData(x=x, y=y)
        line = Plot(plotdata2)
        line.plot(("x", "y"), type="scatter", color="blue")

        container = HPlotContainer(scatter, line)
        #self.plot = container
        self.plot = container


class CSVGrapher(HasTraits):
    """
    主界面包括绘图列表，数据源，文件选择器和添加绘图按钮
    """
    file_folder = Directory() # 文件夹选择
    
    graph_list = List(Instance(Graph)) # 绘图列表
    file_list = Instance(FileList) # 文件列表 
    add_file_list = Button(u"导入文件列表") # 添加绘图按钮

    view = View(
        # 整个窗口分为上下两个部分
        VGroup(
            # 上部分横向放置控件，因此用HGroup
            HGroup(
                # 文件选择控件
                Item("file_folder", label=u"选择存放tif文件的文件夹", width=400),
                # 添加绘图按钮
                Item("add_file_list", show_label=False)
            ),
            # 下部分是绘图列表，采用ListEditor编辑器显示
            Item("graph_list", style="custom", show_label=False, 
                 editor=ListEditor(
                     use_notebook=True, # 是用多标签页格式显示
                     deletable=True, # 可以删除标签页
                     dock_style="tab", # 标签dock样式
                     page_name=".name") # 标题页的文本使用Graph对象的name属性
                )
        ),
        resizable = True,
        height = 0.8,
        width = 0.8,
        title = u"Multifractal Spectrum多分维谱线测量工具"
    )

    def _file_folder_changed(self):
        """
        打开新文件夹时的处理，根据文件夹名称创建一个FileList
        """
        self.file_list = FileList()
        self.file_list.file_name(self.file_folder,'.tif')
        # del self.graph_list[:]

    def _add_file_list_changed(self):
        """
        导入文件列表按钮的事件处理
        """
        if self.file_list != None:
            self.graph_list.append( Graph(file_list = self.file_list) )

Beijing = [
    Graph(
        file_list =FileList(file_list=[""]),
        center_coordinate = np.array([[39687281,4044784]]),
        # Changzhou  495423.0,3517391
        # Beijing: point polygon polygon_buffer 12953728,4851892
        # Weifang: 39687281,4044784
        extent = np.array([[153600,184320]]),
        # Changzhou  50153.,66573.
        # Beijing: point polygon polygon_buffer 45599,40099 | 45056,39936
        # Weifang: 153600,184320
        clip_filefolder = u"/home",
        resample_size = "1024",
        resample_filefolder = u"/home",
        q_range = [[-20,1,20]],
        intercept = "Fix the intercept to zero during OLS",
        export_data = "Export the data to .hdf5 file",
        export_png = "Export the result to .png file",
        effective_scale_min = "1",
        effective_scale_max = "1024",
        export_filefolder = u"/home")
]
                       
if __name__ == "__main__":
    
    csv_grapher = CSVGrapher(
            file_folder="",
            graph_list=Beijing)
    csv_grapher.configure_traits()

