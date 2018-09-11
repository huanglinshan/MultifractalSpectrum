# MultifractalSpectrum
multifractal spectrum calculation with python

# 操作说明：
    1、进入的数据：用地为1，NoData=0。
    2、Clip出来的数据，不能乘以255。2018-08-04修改之后（1.0中依然是乘以255）
    3、0.01对于点数据不适用（1.0的思路是先重采样，然后再计算，因此会抹杀大量点数据）
    4、0.1开始进行修改，以便适合点数据（思路为先读取原始数据，再按照盒子尺寸求每个盒子中的sum）
        保证采样盒子的宽与高是采样像素X分辨率（cell大小）的整数倍，比如1024的整数倍。
        这里resample 的意味其实改变了。没有resample的变形。
        北京的（45056，39936）
    5、名称保留前15位
    6、由于windows和linux中的路径分隔符不同，因此在
        def file_name(self, filefoldername, type):区分了平台

# 更改部分：
	1、注释
	2、图例的符号
	3、检验是否有效
	4、输出文字格式（换行）
	5、运行时间
	6、修改启动的画面
	7、加入图标
	8、加入结束提示声音
	9、让下方谱线的坐标轴固定

# 注意：
    1、输出的 Pq_miu_lnmiu文件都一样（Q=-20的结果）
    2、输出的 parameters_and_calculations文件
