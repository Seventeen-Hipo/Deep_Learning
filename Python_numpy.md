# **numpy**（高效的运算工具）

## 1.1 Numpy的优势

### 1.1.1 Numpy介绍（数值计算库）

​	1）num：numerical 数值化的

​	2）py：python

​	3）Numpy（Numerical Python）是一个开源的Python科学计算库，用于快速处理任意维度的数组 。Numy支持常见的数组和矩阵操作。对于同样的数值计算任务，使用Numpy比直接使用Python要简洁的多

​	4）Numpy使用ndarray对象来处理多维数组，该对象是一个快速而灵活的大数据容器

### 1.1.2 ndarray 介绍

​	1）n：任意个

​	      d：dimension（维度）

​	      array：数组

​	2）Numpy提供了一个N维数组类型ndarray，它描述了相同类型的"items"的集合

| 语文 | 数学 | 英语 | 政治 | 体育 |
| :--: | :--: | :--: | :--: | :--: |
|  80  |  89  |  86  |  67  |  79  |
|  78  |  97  |  89  |  67  |  81  |
|  90  |  94  |  78  |  67  |  74  |
|  91  |  91  |  90  |  67  |  69  |
|  76  |  87  |  75  |  67  |  86  |
|  70  |  79  |  84  |  67  |  84  |
|  94  |  92  |  93  |  67  |  64  |
|  86  |  85  |  83  |  67  |  80  |

​	3）用ndarray进行存储：

```python
import numpy as np

score = np.array([[80,89,86,67,79],[78,97,89,67,81],[90,94,78,67,74],[91,91,90,67,69],[76,87,75,67,86],[70,79,84,67,84],[94,92,93,67,80],[86,85,83,67,80]])
```

​	4）ndarray与Python原生list运算效率的对比

```python
import random
import time
import numpy as np
a = []
for i in range(100000000):
	a.append(random.random())
t1 = time.time()
sum1 = sum(a)
t2 = time.time()

b = np.array(a)
t4 = time.time()
sum3 = np.sum(b)
t5 = time.time()
print(t2-t1,t5-t4)
```

### 1.1.3 ndarray的优势

​	1）存储风格：

​		ndarray：相同类型，通用性不强

​		list：不同类型，通用性强

​	2）并行化运算：

​		ndarray支持并行化运算（向量化运算）

​	3）底层语言：

​		Numpy底层使用C语言编程，内部解除了GIL（全局解释器锁），其对数组的操作速度不受Python解释器的限制，效率远高于纯Python代码

## 1.2 认识N维数组（ndarray的属性）

### 1.2.1  ndarray的属性

​	数组属性反映了数组本身固有的信息只能使用列表表示，不能使用元组

|     属性名字     |      属性解释      |
| :--------------: | :----------------: |
|  ndarray.shape   |   数组维度的元组   |
|   ndarray.ndim   |      数组维度      |
|   ndarray.size   |  数组中的元素数量  |
| ndarray.itemsize | 一个数组元素的长度 |
|  ndarray.dtype   |   数组元素的类型   |

**注意：在ndarray创建时，如果没有指定dtype又是整数的话，则默认是 int64 的类型**

### 1.2.2 ndarray的形状

​	1）创建一些数组

```python
# 创建不同形状的数组
a = np.array([[1,2,3],[4,5,6]]) #(2,3)
b = np.array([1,2,3,4]) #(4,)
c = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]]) #(2,2,3)
```

​	2）既可以使用元组，也可以使用列表来表示形状 

1.2.3 ndarray的类型

​	1）常用的类型

|     名称      |                       描述                        | 简写  |
| :-----------: | :-----------------------------------------------: | :---: |
|    np.bool    |     用一个字节存储的布尔类型（True 或 False）     |  'b'  |
|    np.int8    |              一个字节大小，-128到127              |  'i'  |
|   np.int16    |                整数，-32768到32767                | 'i2'  |
|   np.int32    |                整数，-2^31到2^31-1                | 'i4'  |
|   np.int64    |                整数，-2^63到2^63-1                | 'i8'  |
|   np.uint8    |                无符号整数，0到255                 |  'u'  |
|   np.uint16   |               无符号整数，0到65535                | 'u2'  |
|   np.uint32   |              无符号整数，0到2**32-1               | 'u4'  |
|   np.uint64   |              无符号整数，0到2**63-1               | 'u8'  |
|  np.float16   | 半精度浮点数：16位，正负号1位，指数5位，精度10位  | 'f2'  |
|  np.float32   | 单精度浮点数：32位，正负号1位，指数8位，精度32位  | 'f4'  |
|  np.float64   | 双精度浮点数：64位，正负号1位，指数11位，精度52位 | 'f8'  |
| np.complex64  |     复数，分别用两个32位浮点数表示实部和虚部      | 'c8'  |
| np.complex128 |     复数，分别用两个64位浮点数表示实部和虚部      | 'c16' |
|  np.object_   |                    python对象                     |  'O'  |
|  np.string_   |                      字符串                       |  'S'  |
|  np.unicode_  |                    Unicode类型                    |  'U'  |

​	2）使用语法

```python
# 方法1
dtype = np.float32
# 方法2
dtype = ('float32')
```

### **注意：若不指定，整数默认int64，小数默认float64**

## 1.3 基本操作

### 1.3.1 生成数组的方法

#### 	1）生成0和1的数组

```python
# 方法一：
empty(shape[,dtype,order]) empty_like(a[,dtype,order,subok])
eye(N[,M,k,dtype,oeder])
# 方法二：
identity(n[,dtype])
# 方法三：（推荐）
ones(shape[,dtype,order])
# 方法四：
ones_like(a[,dtype,order,subok])
# 方法五：（推荐）
zeros(shape[,dtype,order]) zeros_like(a[,dtype,order,subok])
full(shape,fill_value[,dtype,order])
# 方法六：
full_like(a,fill_value[,dtype,order,subok])
```

​		（1）演示：

```python
# 生成0和1
# 生成多个0
np.zeros(shape = (3,4),dtype = "float32")
# 生成多个1
np.ones(shape = [2,3],dtype = "int32")
```

#### 	2）从现有的数组中生成

​		（1）np.array() **深拷贝**（常用）

```python
score = np.array([[80,89,86,67,79],[78,97,89,67,81],[90,94,78,67,74],[91,91,90,67,69],[76,87,75,67,86],[70,79,84,67,84],[94,92,93,67,80],[86,85,83,67,80]])
# np.array()
data1 = np.array(score)
```

​		（2）np.copy() **深拷贝**

```python
score = np.array([[80,89,86,67,79],[78,97,89,67,81],[90,94,78,67,74],[91,91,90,67,69],[76,87,75,67,86],[70,79,84,67,84],[94,92,93,67,80],[86,85,83,67,80]])
# np.asarray()
data2 = np.asarray(score)
```

​		（3）np.asarray() **浅拷贝**

```python
score = np.array([[80,89,86,67,79],[78,97,89,67,81],[90,94,78,67,74],[91,91,90,67,69],[76,87,75,67,86],[70,79,84,67,84],[94,92,93,67,80],[86,85,83,67,80]])
# np.copy()
data3 = np.copy(score)
```

#### 	3）生成固定范围的数组

（1）np.linespace(x,y,n)：

​	生成n个x到y之间的数，左闭右闭（[x,y]），等距离

（2）np.arange()：

​	类似于range(a,b,c)，生成a到b之间的数，左闭右开（[a,b)），步长为c

#### 	4）生成随机数组

##### （1）np.random模块

###### 	（a）均匀分布：

​		1、np.random.rand(d0，d1，...，dn)

​			返回[0.0，1.0)内的一组均匀分布的函数

​		2、np.random.uniform(low = 0.0,high = 1.0,size = None)

​			功能：从一个均匀分布[low,high)中随机采样，定义域是左闭右开，即包含low，不包含high。

​			参数介绍：

​				low：采样下界，float类型，默认值为0；

​				high：采样上界，float类型，默认值为1；

​				size：输出样本数目，为int或元组(tuple)，例如，size = (m,n,k)，则输出mnk个样本，缺省时输出1个值

​				返回值：ndarray类型，其形状和参数size中描述的一致。

​		3、np.random.randint(low,high = None,size = None,dtype = 'l')

​			从一个均匀分布中随机采样，生成一个整数或N维整数数组，取数范围：若high不为None时，取[low,high)之间随机整数，否则取值[0,low)之间随机整数。		

​		4、补充：

​			均匀分布（Uniform Distribution）是概率统计中的重要分布之一。顾名思义，均匀表示可能性相等的含义。均匀分布在自然情况下极为罕见，而人工栽培的有一定株行距的植物群落即是均匀分布

###### 	（b）正态分布（理解）：

​		1、什么是正态分布：

​			正态分布是一种概率分布。正态分布是具有两个参数μ和σ的连续型随机变量的分布，第一参数μ是服从正态分布的随机变量的均值，第二个参数σ是此随机变量的方差，所以正态分布记作N(μ，σ)。

​		2、正态分布的应用：

​			生活、生产与科学实验中很多随机变量的概率分布都可以近似的用正态分布来描述。

​		3、正态分布的特点：

​			μ决定了其位置，其标准差σ。决定了分布的幅度。当μ = 0，σ = 1时的正态分布是标准正态分布。

​		4、标准差与方差的意义：

​			标准差越小，离散程度越小；标准差越大，离散程度越大；

> [!IMPORTANT]
>
> ###### 正态分布：
>
> （1）np.random.randn(d1,d2,...,dn)
>
> ​	功能：从标准正态分布中返回一个或多个样本值
>
> （2）**np.random.normal(loc = 0.0,scale = 1.0,size = None)**
>
> ​	loc：float
>
> ​		此概率分布的均值（对应着整个分布的中心centre）
>
> ​	scale：float
>
> ​		此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越廋高）
>
> ​	size：int or tuple ints
>
> ​		输出的shaoe，默认为None，只输出一个值
>
> （3）np.random.standard_normal(size = None)
>
> ​	返回指定形状的标准正态分布的数组

``` python
# 案例：随机生成8只股票2周的交易日涨幅情况
# 获取数据
data = np.random.normal(loc = 0,scale = 1,size = (8.10))
# 获取第一个股票的前三个交易日的涨跌幅数据
data[0,0:3]
```

### 1.3.2 形状修改

*将行换成列，将列换成行*

​	（1）ndarray.reshape(shape)

``` python
data.reshape((10,8))
```

**注意：此方法只是修改了形状，对原有的数据进行了切割，并没有实现对行列的转换**

​	（2）ndarray.resize()

``` python
data.resize(10,8)
```

**注意：没有返回值，对原始的ndarray进行了修改**

​	（3）ndarray.T

``` python
data.T
```

**注意：转置，列变成行，行变成列**

### 1.3.3 类型的修改

​	（1）ndarray.astype(type)

``` python
data.astype("int32")
```

​	（2）ndarray.tostring([order]) 或者 ndarray.tobytes([order])

**转换成byte**

``` python
data.tostring()
```

### 1.3.4 数组去重

​	（1）ndarray.unique

``` python
temp = np.array([[1,2,3,4],[3,4,5,6]])
np.unique(temp)
```

​	（2）先使用flatten方法，将二维变成一维，再使用set()方法，去重（set方法只能操作一维的）

``` python
set(temp.flatten())
```

## 1.4 ndarray的运算

### 1.4.1 逻辑运算

#### 	（1） 布尔索引：

``` python
# 重新生成8只股票10个交易日的涨跌幅数据
stock_change = np.array.normal(loc = 0,scale = 1,size = (8,10))
# 逻辑判断，如果涨跌幅大于0.5就标记为True，否则为False
stock_change > 0.5
# 输出所有大于0.5的数据
stock_change[stock_change > 0.5]
# 让满足数据大于0.5的，改值为1.1
stock_change[stock_change > 0.5] = 1.1
```

####		（2）通用判断函数

​		（a）np.all(布尔值)

​			只要有一个False就返回False，只有全是True才返回True

​		（b）np.any()

​			只要有一个True就返回True，只有全是False才返回False

``` python
# 判断stock_change[0:2,0:5]是否全是上涨的
stock_change[0:2,0:5] > 0

np.all(stock_change[0:2,0:5] > 0)

# 判断前五只股票这段期间是否有上涨
np.any(stock_change[:5,:] > 0)
```

####  	（3）np.where （三元运算符）

``` python
# 判断前4个股票前四天的涨跌幅，大于0的设置为1，否则为0
# np.where(布尔值,True的位置的值,False的位置的值)
temp = stock_change[:4,:4]
np.where(temp > 0,1,0)
```

#### 	(4)复合逻辑需要结合np.logical_and 和 np.logical_or使用

``` python
# 判断前四个股票前四天的涨跌幅，大于0.5并且小于1的，换为1，否则为0
# 判断前四个股票前四天的涨跌幅，大于0.5或者小于-0.5的换为1，否则为0
np.logical_and(temp > 0.5,temp < 1)
np.where(np.logical_and(temp > 0.5,temp < 1),1,0)

np.logical_or(temp > 0.5,temp < -0.5)
np.where(np.logical_or(temp > 0.5,temp < -0.5),1,0)
```

### 1.4.2 统计运算

####  	（1）统计指标函数

> [!IMPORTANT]
>
> ​	1）np.min(a[,axis,out,keepdims])：最小值
>
> ​	2）np.max(a[,axis,out,keepdims])：最大值
>
> ​	3）np.median(a[,axis,out,overwrite_input,keepdims])：中位数
>
> ​	4）np.mean(a[,axis,dtype,out,keepdims])：平均值
>
> ​	5）np.std(a[,axis,dtype,out,ddof,keepdims])：标准差
>
> ​	6）np.var(a[,axis,dtype,out,ddof,keepdims])：方差

> [!NOTE]
>
> 可以使用np.函数名 或者 使用ndarray.方法名 调用

> [!CAUTION]
>
> 进行统计的时候，axis轴的取值并不一定，Numpy中不同的API轴的值都不一样，(axis 0)代表列，(axis 1)代表行，去进行统计

``` python
# 前四只股票前四天的最大涨幅
temp.max(axis = 1)

np.max(temp,axis = 1)
# 前四只股票前四天的最大跌幅
np.min(temp,axis = 1)
# 前四只股票前四天的波动程度
np.std(temp,axis = 1)
# 前四只股票前四天的平均涨跌幅
np.mean(temp,axis = 1)
```

####  	（2）返回最大值、最小值的坐在位置

​		1）np.argmax(temp,axis = )

​		2）np.argmin(temp,axis = )

### 1.4.3 数组间运算

#### 	（1）数组与数的运算

``` python
arr = np.array([[1,2,3,2,1,4],[5,6,1,2,3,1]])

arr + 1
arr / 2
```

#### 	（2）广播机制

执行 broadcast 的前提在于，两个 ndarray 执行的是 element-wish 的运算，Broadcast机制的功能是为了方便不同形状的 ndarray（numpy库的核心数据结构）进行数学运算

当操作两个数组时，numpy会逐个比较它们的shape（构成的元组tuple），只有在下述情况下，两个数组才能够进行数组与数组的运算

> [!IMPORTANT]
>
> ​	1）维度相等
>
> ​	2）shape（其中相对应的一个地方为1）

例如：

``` python
Image	(3d array):	256 x 256 x 3
Scale	(1d array):	            3
Result	(3d array):	256 x 256 x 3

A		(4d array):	9 x 1 x 7 x 1
B		(3d array):	    8 x 1 x 5
Result	(4d array):	9 x 8 x 7 x 5

A		(2d array):	5 x 4
B		(1d array):	    1
Result	 (2d array): 5 x 4

A		(3d array):	15 x 3 x 5
B		(3d array):	15 x 1 x 1
Result	 (3d array): 15 x 3 x 5
```

如果是下面这样，则不匹配：

``` python
A	(1d array):	10
B	(1d array):	12

A	(2d array):		2 x 1
B	(3d array):	8 x 4 x 3
```

### 1.4.4 矩阵运算

#### 	1）什么是矩阵：

矩阵（matrix），和array的区别，矩阵必须是二维的，但是array可以是多维的

#### 	2）存储矩阵

``` python
# ndarray 存储矩阵
a = np.array([[80,86],
             [82,80],
             [85,78],
             [90,90],
             [86,82],
             [82,90],
             [78,80],
             [92,94]])
# matrix 存储矩阵
b = np.mat([[80,86],
             [82,80],
             [85,78],
             [90,90],
             [86,82],
             [82,90],
             [78,80],
             [92,94]])
```

#### 	3）np.mat()：

​	将数组转换为矩阵类型

```python
a = np.array([[80,86],
             [82,80],
             [85,78],
             [90,90],
             [86,82],
             [82,90],
             [78,80],
             [92,94]])
b = np.array([[0,3],[0,7]])

a_mat = np.mat(a)
b_mat = np.mat(b)
```

#### 	4）矩阵乘法运算：

​		（1）用 ndarry 存储矩阵的方法：

​			（a）np.matmul

​			（b）np.dot

​			（c）运算符：@

``` python
# np.matmul
np.matmul(a,b)
# np.dot
np.dot(a,b)
# @ 运算符法
a @ b
```

 		（2）用 matrix 存储矩阵的方法：

``` python
a_mat * b_mat
```

## 1.5 合并与分割

### 1.5.1 合并

​	1）numpy.hstack(tup)：水平拼接

​	2）numpy.vstack(tup)：竖直拼接

​	3）numpy.concatenate((a1,a2,....),axis = 0)：自定义拼接；水平拼接axis = 1，竖直拼接axis = 0（默认）

``` python
a = np.array((1,2,3))
b = np.array((2,3,4))
np.hstack((a,b))
# array([1,2,3,2,3,4])
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.hstack((a,b))
# array([[1,2],[2,3],[3,4]])
```

``` python
a = np.array([1,2,3])
b = np.array([2,3,4])
np.vstack((a,b))
# array([[1,2,3],[2,3,4]])
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.vstack((a,b))
# array([[1],[2],[3],[2],[3],[4]])
```

### 1.5.2 分割

​	1）numpy.split(ary,indices_or_sections,axis = 0)

``` python
x = np.arange(9.0)
# array([0.,1.,2.,3.,4.,5.,6.,7.,8.])
np.split(x,3)
# [array([0.,1.,2.]),array([3.,4.,5.]),array([6.,7.,8.])]
x = np.arange(8.0)
# array([0.,1.,2.,3.,4.,5.,6.,7.])
np.split(x,[3,5,6,10])
# [array([0.,1.,2.]),array([3.,4.]),array([5.]),array([6.,7.]),array([],dtype = float64)]
```

## 1.6 IO操作和数据处理

### 1.6.1 Numpy读取

​	1）genfromtxt(fname[,dtype,comments,...])

``` python
# 读取数据
test = np.genfromtxt("路径",delimiter = '分隔符')
```

> [!WARNING]
>
> 一般不使用Numpy读取，字符串会出现无法读入的问题；nan 则是表示数据缺失，nan的类型是float64

### 1.6.2 如何处理缺失值

​	1）直接删除含有缺失值的样本

​	2）替换、插补
