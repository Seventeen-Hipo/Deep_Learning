# **matplotlib**

## 1.1 什么是Matplotlib

1、专门用于开发2D图表（包括3D图表）

2、使用起来及其简单

3、以渐进、交互式方式实现数据可视化

4、补充;

​	1) mat - （matrix）矩阵 - 二维数据 - 二维图表

​	2) plot - 画图

​	3) lib - library 库

## 1.2 为什么使用matplotlib

1）能将数据进行可视化，更直观的呈现

## 1.3 matplotlib的三层结构

​	1）容器层：

​		画板层（Canvas）

​		画布层（Figure）：plt.figure()

​		绘图区/坐标系（axes）：plt.subplots()

​	2）辅助显示层：

​	3）图像层：

## 1.4 折线图

### 1.4.1 折线图绘制与保存图片

#### 	1）matplotlib.pyplot模块

​		matplotlib.pyplot包含了一系列类似于matlab的画图函数。它的函数作用于当前图形（figure）的当前坐标系（axes）

```python
import matplotlib.pyplot as plt
```

#### 	2) 折线图绘制与显示

​		展示某地一周的天气

```python
# 导包
import matplotlib.pyplot as plt
# 1）创建画布（容器层）
plt.figure()
# 2）绘制折线图（图像层）
plt.plot([1,2,3,4,5,6,7],[17,17,18,15,11,11,13])
# 3）显示图像
plt.show()
```

#### 	3)  设置画布属性与图片保存

```python
plt.figure(figsize = (),dpi = )
# figsize：指定图的长宽
# dpi：图像的清晰度
# 返回fig对象
plt.savefig(path)
```

 **注意：plt.show()会释放figure资源，如果在显示图像之后保存图片将只能保存空图片**

### 1.4.2 完善原始折线图 1（辅助显示层）

```python
# 需求：画出某城市11点到12点，1小时内每分钟的温度变化折线图，温度范围在15度~18度
import random
# 准备数据
x = range(60)
y_shanghai = [random.uniform(15,18) for i in x]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制图像
plt.plot(x,y_shanghai)
# 显示图像
plt.show()
```

#### 	1) 添加自定义x,y刻度

​	（1）plt.xticks(x,**kwargs)

​		x：要显示的刻度值

​		**kwargs：对x的说明

​	（2）plt.yticks(y,**kwargs)

​		y：要显示的刻度值

​		**kwargs：对y的说明

```python
# 需求：画出某城市11点到12点，1小时内每分钟的温度变化折线图，温度范围在15度~18度
import random
# 准备数据
x = range(60)
y_shanghai = [random.uniform(15,18) for i in x]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制图像
plt.plot(x,y_shanghai)
# 准备x的刻度说明
x_label = ["11点{}分".format(i) for i in x]
# 修改x，y的刻度值
plt.xticks(x[::5],x_label[::5])
plt.yticks(range(0,40,5))
# 显示图像
plt.show()
```

#### **注意：存在中文显示问题，需要加入下列代码**

```python
plt.rcParams['font.sans-serif'] = ['SimHei',]
plt.rcParams['axes.unicode_minus'] = False
```

#### 2）添加网格显示

```python
plt.grid(True,linestyle = "--",alpha = 0.5)
```

​	（1）linestyle：线条风格

​	（2）alpha：透明度

#### 3）添加描述信息

```python
plt.xlabel("x的描述信息")
plt.ylabel("y的描述信息")
plt.title("标题")
```

### 1.4.3 完善原始折线图2（图像层）

```python
# 需求：再添加一个城市的温度变化
# 收集到北京当天温度变化情况，温度再1度到3度
# 准备北京数据
y_biejing = [random.uniform(1,3) for i in x]
```

修改后的代码：

```python
# 导包
import random
# 需求：再添加一个城市的温度变化
# 收集到北京当天温度变化情况，温度再1度到3度
y_biejing = [random.uniform(1,3) for i in x]
# 准备数据
x = range(60)
y_shanghai = [random.uniform(15,18) for i in x]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制图像
plt.plot(x,y_shanghai)
plt.plot(x,y_beijing)
# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei',]
plt.rcParams['axes.unicode_minus'] = False
# 准备x的刻度说明
x_label = ["11点{}分".format(i) for i in x]
# 修改x，y的刻度值
plt.xticks(x[::5],x_label[::5])
plt.yticks(range(0,40,5))
# 显示图像
plt.show()
```

#### 	1）修改线条的颜色、风格

```python
plt.plot(x,y_shanghai,color = "r",linestyle = "--")
plt.plot(x,y_beijing,color = "b")
```

#### 	2）对应的线条颜色和风格

| 颜色字符  |     风格字符     |
| :-------: | :--------------: |
| 红色（r） |    实线（-）     |
| 绿色（g） |    虚线（--）    |
| 蓝色（b） |   点划线（-.）   |
| 白色（w） |   点虚线（：）   |
| 青色（c） | 留空、空格（''） |
| 洋红（m） |                  |
| 黄色（y） |                  |
| 黑色（k） |                  |

#### 	3）图例的显示

​	既要在图像层中修改，也要在辅助显示层上进行修改

#### 		（1）在图像层中修改

```python
plt.plot(x,y_shanghai,color = "r",linestyle = "--",label = "上海")
plt.plot(x,y_beijing,color = "b",label = "北京")
```



#### 		（2）在辅助显示层显示图例

```python
plt.legend()
```

​	图例默认是：plt.legend(loc = "best")，可以在括号内修改默认值

| Location String | Location Code |
| :-------------: | :-----------: |
|     'best'      |       0       |
|  'upper right'  |       1       |
|  'upper left'   |       2       |
|  'lower right'  |       3       |
|  'lower left'   |       4       |
|     'right'     |       5       |
|  'center left'  |       6       |

​	例如：plt.legend(loc = "lower left") 或者是 plt.legend(loc = 4)

### 1.4.4 多个坐标系显示-plt.subplots（面向对象的画图方法）

​	**可以通过subplots函数实现**

​	1）matplotlib.pyplot.subplot(nrows = 1,ncols = 1,**fig_kw) 创建一个带有多个axes（坐标系/绘图区）的图	

​		nrows：多少行

​		 ncols：多少列

​		调用该函数会返回：figure（画布），axes（绘图区）

**注意：plt.函数名()相当于面向过程的画图方法，axes.set_方法名()相当于面向对象的画图方法** 

```python
# 画出温度变化图
import random
import matplotlib.pyplot as plt
# 准备x，y坐标的数据
x = range(60)
y_shanghai = [random.uniform(15,18) for i in x]
# 增加北京的温度数据
y_beijing = [random.uniform(1,3) for i in x]
# 创建画布
# plt.figure(figsize = (20,8),dpi = 80)
figure,axes = plt.subplots(nrows = 1,ncols = 2,figsize = (20,8),dpi = 80)
# 绘制折线图
# plt.plot(x,y_shanghai)
# plt.plot(x,y_shanghai,label = '上海')
axes[0].plot(x,y_shanghai,label = '上海')
# 使用多次plot可以画多个折线
# plt.plot(x,y_biejing,color = 'r',linestyle = '--',label = '北京')
axes[1].plot(x,y_beijing,color = 'r',linestyle = '--',label = '北京')
# 显示图例
# plt.legend(loc = 'best')
axes[0].legend()
axes[1].legend()
# 增加以下两行代码
# 构造x轴刻度标签
x_ticks_label = ["11点{}分".format(i) for i in x]
# 构造y轴刻度
y_ticks = range(40)
# 修改x，y轴坐标的刻度显示
# plt.xticks(x[::5],x_ticks_label[::5])
# plt.yticks(y_ticks[::5])
axes[0].set_xticks(x[::5],x_ticks_label[::5])
axes[0].set_yticks(y_ticks[::5])
axes[1].set_xticks(x[::5],x_ticks_label[::5])
axes[1].set_yticks(y_ticks[::5])
# 添加网格显示
# plt.grid(True,linestyle = '--',alpha = 0.5)
axes[0].grid(True,linestyle = '--',alpha = 0.5)
axes[1].grid(True,linestyle = '--',alpha = 0.5)
# 添加描述信息
# plt.xlabel("时间变化")
# plt.ylabel("温度变化")
# plt.title("上海、北京11点到12点每分钟的温度变化状况")
axes[0].set_xlabel("时间变化")
axes[0].set_ylabel("温度变化")
axes[0].set_title("上海11点到12点每分钟的温度变化状况")
axes[1].set_xlabel("时间变化")
axes[1].set_ylabel("温度变化")
axes[1].set_title("北京11点到12点每分钟的温度变化状况")
# 中文显示问题修改
plt.rcParams['font.sans-serif'] = ['SimHei',]
plt.rcParams['axes.unicode_minus'] = False
# 显示图
plt.show()
```

### 1.4.5 折线图的应用场景（拓展）

​	1）画各种数学函数的图像

​		（1）正弦函数图像

```python
import numpy as np
# 准备数据
x = np.linspace(-10,10,1000)
y = np.sin(x)
# 创建画布
plt.figure(figsize = (20,8),dpi = 100)
# 绘制函数图像
plt.plot(x,y)
# 添加网格显示
plt.grid()
# 显示图像
plt.show()
```



## 1.5 散点图（scatter）

### 1.5.1  常见图形的种类及意义

#### 	1）折线图：

​		（1）以折线的上升或下降来表示统计数量的增减变化的统计图

​		（2）特点：能够显示数据的变化趋势，反映事物的变化情况

#### 	2）散点图：

​		（1）用两组数据构成多个坐标点。考察坐标点的分布情况，判断两变量之间是否存在某种关联或总结坐标点的分布模式

​		（2）特点：判断变量之间是否存在数量关联趋势，展示离群点（关系/规律）

#### 	3）柱状图：

​		（1）排列在工作表的列或行中的数据可以绘制到柱状图中

​		（2）特点：绘制连离散的数据，能够一眼看出各个数据的大小，比较数据之间的差别（对比/统计）

#### 	4）直方图：

​		（1）由一系列高度不相等的纵向条纹或线段表示数据分布情况。一般用横轴表示数据范围，纵轴表示分布情况

​		（2）特点：绘制连续性的数据展示一组或者多组数据的分布情况（统计）

#### 	5）饼图：

​		（1）用于表示不同分类的占比情况，通过弧度大小来对比各种分类

​		（2）特点：分类数据的占比情况（占比）

### 1.5.2 散点图的绘制

需求：探究房屋面积和房屋价格之间的关系

1）房屋面积

```python
x = [225.98,247.07,253.14,457.85,241.58,301.01,20.67,288.64,163.56,120.06,207.83,342.75,147.9,53.06,224.72,29.51,21.61,483.21,245.25,399.25,343.35]
```

2）房屋价格

```python
y = [196.63,203.88,210.75,372.74,202.41,247.61,24.9,239.34,140.32,104.15,176.84,288.23,128.79,49.64,191.74,33.1,30.74,400.02,205.35,330.64,283.45]
```

3）代码

```python
# 准备数据
x = [225.98,247.07,253.14,457.85,241.58,301.01,20.67,288.64,163.56,120.06,207.83,342.75,147.9,53.06,224.72,29.51,21.61,483.21,245.25,399.25,343.35]
y = [196.63,203.88,210.75,372.74,202.41,247.61,24.9,239.34,140.32,104.15,176.84,288.23,128.79,49.64,191.74,33.1,30.74,400.02,205.35,330.64,283.45]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制图像
plt.scatter(x,y)
# 显示图像
plt.show()
```



## 1.6 柱状图（bar）

### 1.6.1 柱状图的绘制

​	1）需求1-对比每部电影的票房收入

​		（1）准备数据	

```python
movie_name = ['雷神3','诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴','降魔案','追捕','七十七天','密战','狂兽','其他']
tickets = [73853,57767,22354,15969,14839,8725,8716,8318,7916,8318,6764,52222]
```

​		（2）绘制

```python
matplotlib.pyplot.bar(x,y,width,align = 'center',**kwargs)
```

​			x：有几个类别（横坐标）

​			y：对应的票房数量

​			width：柱状图的宽度

​		（3）代码

```python
# 准备数据
movie_names = ['雷神3','诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴','降魔案','追捕','七十七天','密战','狂兽','其他']
tickets = [73853,57767,22354,15969,14839,8725,8716,8318,7916,8318,6764,52222]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制图像
x_ticks = range(len(movie_names))
plt.bar(x_ticks,tickets,color = ['b','r','g','y','c','m','y','k','c','g','b'])
# 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei',]
plt.rcParams['axes.unicode_minus'] = False
# 修改刻度
plt.xticks(x_ticks,movie_names)
# 添加标题
pit.title("电影票房收入对比")
# 添加网格显示
plt.grid(linestyle = '--',alpha = 0.5)
# 显示图像
plt.show()
```

​	1）需求2-对比每部电影的票房收入更能有说服力

​		（1）准备数据

```python
movie_name = ['雷神3','诸神黄昏','正义联盟','寻梦环游记']
first_day = [10587.6,10062.5,1275.7]
first_weekend = [36224.9,34479.6,11830]
```

​		（2）代码

```python
# 准备数据
movie_name = ['雷神3','诸神黄昏','正义联盟']
first_day = [10587.6,10062.5,1275.7]
first_weekend = [36224.9,34479.6,11830]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制柱状图
plt.bar(range(3),first_day,width = 0.2,label = '首日票房')
plt.bar([i+0.2 for i in range(3)],first_weekend,width = 0.2,label = '首周票房')
# 显示图例
plt.legend()
# 修改刻度
plt.xticks([i + 0.1 for i in range(3)],movie_name)
# 显示图像
plt.show()
```



## 1.7 直方图（histogram）

### 1.7.1 相关概念

​	1）组数：在统计数据时，我们把数据按照不同的范围分成几组，分成的组的个数称为组数

​	2）组距：每一组两个端点的差

### 1.7.2  直方图的绘制

​	1）需求：电影时长的分布情况

​	2）准备数据：

```python
time = [15,12,17,19,20,11]
```

​	3）绘制：

```python
matplotlib.pyplot.hist(x,bins = None,normed = None,**kwargs)
```

​		（1）x：本需求中是time

​		（2）bins：组数 = （max(time)-min(time)）/组数

​		（3）density（normed）：是否显示频率，默认是不显示，normed 参数可能会报过时错误

​	4）代码：

```python
# 准备数据
time = [15,12,17,19,20,11]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制直方图
distance = 2
group_num = int((max(time)-min(time)) / distance)
plt.hist(time,bins = group_num)
# 修改x轴刻度
plt.xticks(range(min(time),max(time) + 2,distance))
# 添加网格
plt.grid(linestyle = '--',alpha = 0.5)
# 显示图像
plt.show()
```



## 1.8 饼图（pie）

### 1.8.1  绘制饼图

​	1）准备数据：

```python
movie_name = ['雷神3','诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴','降魔案','追捕','七十七天','密战','狂兽','其他']
place_count = [73853,57767,22354,15969,14839,8725,8716,8318,7916,8318,6764,52222]
```

​	2）绘制

```python
plt.pie(x,labels = ,autopct = ,colors)
```

​		（1）x：数量，自动算百分比

​		（2）labels：每部分名称

​		（3）autopct：占比显示指定%1.2f%%

​		（4）colors：每个部分的颜色

​	3）代码

```python
# 准备数据
movie_name = ['雷神3','诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴','降魔案','追捕','七十七天','密战','狂兽','其他']
place_count = [73853,57767,22354,15969,14839,8725,8716,8318,7916,8318,6764,52222]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制饼图
plt.pie(place_count,labels = movie_name,autopct = '%1.2f%%',colors = ['b','r','g','y','c','m','y','k','c','g','y','g'])
# 显示图例
plt.legend()
# 显示图像
plt.show()
```

​	4）完善：

​		（1）添加axis：为了让显示的饼图保持圆形，需要添加axis保证长宽一样

```python
plt.axis('equal')
```

​	5）最终代码：

```python
# 准备数据
movie_name = ['雷神3','诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴','降魔案','追捕','七十七天','密战','狂兽','其他']
place_count = [73853,57767,22354,15969,14839,8725,8716,8318,7916,8318,6764,52222]
# 创建画布
plt.figure(figsize = (20,8),dpi = 80)
# 绘制饼图
plt.pie(place_count,labels = movie_name,autopct = '%1.2f%%',colors = ['b','r','g','y','c','m','y','k','c','g','y','g'])
# 显示图例
plt.legend()

plt.axis('equal')

# 显示图像
plt.show()
```
