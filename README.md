# MultifractalSpectrum
multifractal spectrum calculation with python 

### 1、文件说明
    * main            - 主运行文件：定制参数设置
    * fileList        - 存放：生成FileList的类(tif格式文件)
    * clipRaster      - 存放：切割文件的类（按照左上角、右下角坐标设定的矩形切割图片 - 可选项）
    * resizeRaster    - 存放：重采样的类（由于盒子法限制，需要是2的倍数）
    * calculation     - 存放：使用最小二乘法回归拟合参数的类
    * saveDataAndPlot - 存放：保存数据和图像的类
### 2、使用说明
    * 打开main.py。自定义参数（文件目录等）
    * NoData=0
    * 运行文件
    * 查看结果
### 3、预先需要安装
* 下载安装anaconda 最新版（64位 - python3.6）
    > 下载地址：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.0-Windows-x86_64.exe 
* 在anaconda shell中，安装 gdal(命令如下)
    > conda install gdal
    
   安装很慢，需要耐心...各个电脑情况不同，很容易安装遇到问题...

### 4、更新历史
* 2018-11-17:
 1. 添加了读取hdf5文件的程序：extractDataFromH5.py
 2. 修改saveDataAndPlot，使得以下目标可以实现：
    ##### 设置是否进行以下保存
        ### Set whether to save specific part
        # self.if_save_group1 MUST be set True
        # global and local parameters in OLS
        self.if_save_group1 = True
        # self.if_save_group2 recommended to be True
        # parameters_calculation
        self.if_save_group2 = True
        # self.if_save_group3 recommended to be False
        # Pq_miu_lnmiu
        self.if_save_group3 = False
        # self.if_save_group4 recommended to be True
        # P_and_lnP
        self.if_save_group4 = True
3. 可以调整 D alpha 和 f 计算的 标度区（可以取不一致）
    ###
       # 6. set the scaling range for different parameter
        ## scaling range for D_q and tau(q) during OLS estimation
        # minimum: 1
        self.dq_scale_min = 1
        # maximum
        self.dq_scale_max = 512
        dq_scale_ul = np.log2(self.dq_scale_max)
        dq_scale_dl = np.log2(self.dq_scale_min)

        ## scaling range for alpha(q) during OLS estimation
        # minimus: 1
        self.alpha_scale_min = 1
        # maximum
        self.alpha_scale_max = 512
        alpha_scale_ul = np.log2(self.alpha_scale_max)
        alpha_scale_dl = np.log2(self.alpha_scale_min)

        ## scaling range for alpha(q) during OLS estimation
        # minimus: 1
        self.f_scale_min = 1
        # maximum
        self.f_scale_max = 512
        f_scale_ul = np.log2(self.f_scale_max)
        f_scale_dl = np.log2(self.f_scale_min)

        # combined scaling range (would be distinguished by "label")
        self.scale_limit = [(dq_scale_dl, dq_scale_ul), (alpha_scale_dl, alpha_scale_ul), (f_scale_dl, f_scale_ul)]
* 2018-12-06
1. 更改了 calculating 时的 屏幕输出
    ###
        print("Calculating..." + file_name.split('\\')[-1])
2. 不再保留 miu(q) 的矩阵，释放内存，以下代码被注释掉了
    ###
        # miu_q_list_list.append(miu_q_list)

### 5、新的分支（未放入）
    ## 2018-12-01
    1. 使用 Legendre 方法计算 多分维谱，结果一致


