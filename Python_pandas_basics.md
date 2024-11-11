# **Pandas 基础处理**（数据处理工具）

## 1.1 Pandas介绍

​	pandas = panel + data + analysis

​	1）panel：面板数据，用于存储三维的数据，经常使用在计量经济学中

### 1.1.1 Pandas 的三大核心数据结构

#### 	1）DateFrame：

##### 		（1）结构：既有行索引又有列索引的二维数组

``` python
# 创建一个符合正态分布的10个股票5天的涨跌幅数据
stock_change = np.random.normal(0,1,(10,5))
"""
array([[-0.68901805  0.35501262  0.68765585  0.56556047 -0.11998448]
 [ 0.62062716  0.78745885 -0.27263601 -0.50624818  0.63878418]
 [ 2.26821388 -0.2389337   0.78803186 -0.31132509  1.13653041]
 [-0.50798802 -1.46512229  0.78703809 -0.04282486 -0.49810702]
 [ 0.8228952   0.72301001  0.18526914 -0.28830925 -0.82628657]
 [ 0.51857834 -0.9303888   0.53517538 -0.97068582 -0.70211538]
 [ 2.1538919   0.01829773  1.58561784 -0.05337342  0.35944926]
 [-0.02930612 -1.09484876  0.7337958  -0.4201924   1.72148383]
 [-0.73536152 -0.91287463  1.40081363 -1.17861434 -0.0690067 ]
 [-0.87272156  2.11377658  0.99223047  1.87340739 -0.43012044]])
"""
```

这样的数据形式很难看到存储的是什么样的数据，并且也很难获取相应的数据，比如需要获取某个指定股票的数据，就很难去获取

~~~ python
# 使用 Pandas 中的数据结构
stock_change = pd.DataFrame(stock_change)
~~~

``` python
# 添加行索引
stock = ["股票{}".format(i) for i in range(10)]
pd.DataFrame(stock_change,index = stock)
# 添加列索引
data = pd.data_range(start = '20230101',periods = 5,freq = 'B')
a = pd.DataFrame(stock_change,index = stock,columns = data)
```

##### 		（2）属性：

​			（a）shape

​			（b）index

​			（c）columns

​			（d）values

​			（e）T

``` python

# a的形状
a.shape # (10,5)
# a的行索引
a.index
# a的列索引
a.columns
# a刨除了行列索引的值
a.values
# a的转置
a.T
```

##### 		（3）方法：

​			（a）head()：默认返回前5行

​			（b）tail()：默认返回后五行

##### 		（4）DataFrame索引的设置

​			（a）修改行列索引值

> [!WARNING]
>
> 想要修改Pandas中的索引值的时候，只能整体进行修改，不能单独进行修改索引

​			（b）重设索引

~~~ python
# reset_index(drop = False)
# 设置新的下标索引
# drop：默认为False，不删除原来的索引；如果为True，删除原来的索引值
~~~

​			（c）设置新索引（set_index(keys,drop = True)）

​				keys：列索引名或者列索引名称列表

​				drop：boolean，default，True当做新的索引，删除原来的列

``` python
df = pd.DataFrame({'month':[1,4,7,10],'year':[2012,2014,2013,2014],'sale':[55,40,84,31]})
# 将月份设置为新的索引
df.set_index('month',drop = True)
# 设置多个索引
new_df = set_index(['year','month'])
```

#### 	2）MultiIndex 与 Panel：

#### 	（1）MultiIndex：多级或分层索引对象

​		index 的属性：

​			names：levels的名称

​			levels：每个levels的元组值

~~~ python
df_index.names
# FrozenList(['year','month'])
df_index.levels
# FrozenList([[2012,2013,2014],[1,4,7,10]])
~~~

#### 	（2）Panel：（目前已经在pandas库中被移除）

``` python
# Panel的api
class pandas.Panel(data = None,items = None,major_axis = None,minor_axis = None,copy = False,dtype = None)
```

​		（a）存储 3维数组的Panel结构

``` python
p = pd.Panel(np.arange(24).reshape(4,3,2),items = list('ABCD'),major_axis = pd.data_range('20130101',periods = 3),minor_axis = ['first','second'])

"""
<class 'pandas.core.panel.Panel'>
Dimensions: 4 (items) x 3 (major_axis) x 2 (minor_axis)
Items axis: A to D
Major_axis axis: 2013-01-01 00:00:00 to 2013-01-03 00:00:00
Minor_axis axis: first to second
"""
```

> [!CAUTION]
>
> 注：Pandas从版本0.20.0开始弃用Panel。推荐的用于表示3D数据的方法是DataFrame上的MultiIndex方法

#### 	3）Series：（带索引的一维数组）

​	series的结构只有行索引

#### 		（1）属性

​			index：

​			values：

#### 		（2）方法

​		（a）指定内容，默认索引

``` python
pd.Series(np.arange(10))
```

​		（b）指定索引（添加index参数，指定索引值）

~~~ python
pd.Series([6.7,5.6,3,10,2],index = [1,2,3,4,5])
~~~

​		（c）通过字典数据创建

``` python
pd.Series({'red':100,'blue':200,'green':500,'yellow':1000})
```

## 1.2 Pandas的基本数据操作

### 1.2.1 索引操作

``` python
# 读取文件的数据
data = pd.raad_csv("文件路径")
# 删除一些列，让数据更简单些，再去做后面的操作
data = data.drop(["要删除的字段"],axis = 1) # axis = 1是按列删除；axis = 0是按行删除
```

​	1）直接索引：

``` python
# data[1,1] 不能直接进行数字索引，会报错
# 直接索引
data["列索引"]["行索引"] # 必须先列后行
```

​	2）按名字索引：

``` python
# 结合loc或者iloc使用索引
# 使用loc：只能指定行列索引的名字
data.loc['行/列索引','列/行索引'] # data.loc['行/列索引']['列/行索引'] 这种方法也是可行的
```

​	3）按数字索引：

``` python
#使用iloc可以通过索引的下标去获取
data.iloc[1,2]
```

​	4）组合索引：

``` python
# 使用ix组合索引
# 使用ix进行下表和名称组合做索引
data.ix[0:4,['名称']]

# 推荐使用loc和iloc来获取的方式
data.loc[data.index[0:4],['名称']]
data.iloc[0:4,data.columns.get_indexer(['名称'])]
```

### 1.2.2 赋值操作

​	通过上述提到的索引方式，索引到相对应的数据，后通过直接赋值的方式，进行赋值操作·

### 1.2.3 排序

​	1）对内容进行排序：

​	（1）DataFrame

``` python
# 使用 df.sort_values(key = ,ascending = )对内容进行排序
# 单个键或者多个键进行排序，默认升序
# ascending = False：降序
# ascending = True：升序
data = data.sort_values(by = '依据的名称',ascengding = False).head() # 对单个键进行索引

data = data.sort_values(by = ['先依据的名称','若相等再依据的名称'],ascending = False) # 对多个键进行索引
```

​	（2）Series

``` python
# 先获取一个Series
sr = data["名称"]
# 排序
sr.sort_values(ascending = False)
```

​	2）对索引进行排序：

​	（1）DataFrame：

``` python
# 对索引进行排序
data.sort_index()
```

​	（2）Series:

``` python
sr.sort_index()
```

## 1.3 DataFrame的运算

### 1.3.1  算术运算。

``` python
# add(other)：加
data['名称'].add(1)

# sub(other)：减
a = data['start']
b = data['end']
data['a-b'] = a.sub(b)
```

### 1.3.2 逻辑运算（<、>、|、&）

​	1）逻辑运算符

``` python
data['名称'] > 2 # 返回逻辑结果
data[data['名称'] > 2].head() # 逻辑判断的结果作为筛选的依据，返回满足逻辑的结果

# 完成一个或者多个的逻辑判断
(data["名称1"] > 2) & (data["名称2"] > 10)
data[(data["名称1"] > 2) & (data["名称2"] > 10)]
```

​	2）逻辑运算函数

``` python
# query(expr):	expr表示要查询的字符串
data.query("名称1 > 2 & 名称2 > 10")

# isin(values)：判断其中是否存在某些值
data["名称1"].isin([值1,值2]) # 返回逻辑结果
data[data["名称1"].isin([值1,值2])] # 返回满足逻辑的数据
```

### 1.3.3 统计运算

​	1）常规的统计

``` python
# min(最小值),max(最大值),mean(均值),median(中位数),var(方差),std(标准差)
data.max() # 默认是max(axis = 0) 按列求最大值

# describe()方法：放回需要的所有的统计运算
data.describe()

# idxmax(),idxmin()：返回最大值、最小值所在的位置
data.idxmax() # 默认是idxmax(axis = 0) 按列求最大值
```

​	2）累计统计函数

|  函数   |             作用              |
| :-----: | :---------------------------: |
| cumsum  |   计算前1/2/3/.../n个数的和   |
| cummax  | 计算前1/2/3/.../n个数的最大值 |
| cummin  | 计算前1/2/3/.../n个数的最小值 |
| cumprod |   计算前1/2/3/.../n个数的积   |

### 1.3.4 自定义运算

​	apply(func,axis = 0)

​		func：自定义函数

​		axis=0：默认是列；axis=1为行进行运算

``` python
# 定义一个队列，最大值-最小值的函数
data[['名称1','名称2']].apply(lambda x: x.max() - x.min(),axis = 0)
```

## 1.4 Pandas画图

### 1.4.1 pandas.DataFrame.plot

1）DataFrame.plot(x = None,y = None,kind = 'line')

​	（1）x：label or position,default None

​	（2）y：labe;,position or list of label,positions,default None

​	（3）kind：str

​		'line'：折线图（默认）

​		'bar'：竖直柱状图

​		'barh'：水平柱状图

​		'hist'：直方图

​		'pie'：饼图

​		'scatter'：散点图

``` python
data.plot(x = "横坐标",y = "纵坐标",kind = "图的类型")
```

### 1.4.2 pandas.Series.plot

## 1.5 文件读取与存储

### 1.5.1 CSV

#### （1）读取CSV文件 -- read_csv()

​	1）pandas.read_csv(filepath_or_buffer,sep = ',',delimiter = None)

​		filepath_or_buffer：文件路径

​		usecols：指定读取的列名，列表形式

> [!CAUTION]
>
> 如果读取的CSV文件没有字段，要自己手动的添加字段：
>
> pandas.read_csv("文件路径",name = ['要添加的字段'])

#### （2）写入CSV文件 -- to_csv()

​	1）dataframe.to_csv(path_or_buf = None,sep = ',',columns = None,header = True,index = True,index_label = None,mode = 'w',encoding = None)

​		path_or_buf：string or file handle，default None

​		sep：character，default ‘，’

​		columns：sequence，optional

​		mode：'w'：重写；'a'：追加

​		index：是否写进 行索引

​		header：boolean or list of string,default True，是否写进 列索引

​	2）Series.to_csv （path = None,index = True,sep = ',',na_rep = ' ',float_format = None,header = False,index_label = None,mode = 'w',encoding = None,compression = None,data_format = None,decimal = '.'）

``` python
data[:10].to_excel("(自己命名，例如：test.xlsx)",columns = ["要读取的列"],index=False)
```

### 1.5.2 HDF5

​	1、HDF5文件的读取和存储需要指定一个键，值为要存储的DataFrame（存储三维数据的文件）

​	（1）pandas.read_hdf(path_or_buf,key = None,**kwargs)：从h5文件中读取数据

​		path_or_buffer：文件路径

​		key：读取的键

​		mode：打开文件的模式

​		return：Theselected object

​	（2）DataFrame.to_hdf(path_or_buf,key,*kwargs*)

``` python
data = pd.read_hdf("文件路径") # 可以不指定key；但是有多个键值的时候，则必须声明key值
data.to_hdf("要保存到的位置",key = "键值") # 必须要指定key，否则会报错
```

​	2、优先选择使用HDF5文件存储：

​	1）HDF5存储是支持压缩的，使用的方式是blosc，这个是速度最快的也是pandas默认支持的

​	2）使用压缩可以提高磁盘利用率，节省空间

​	3）HDF5还是跨平台的，可以轻松迁移到hadoop上面

### 1.5.3 JSON

​	（1）pandas.read_json(path_or_buf = None,orient = None,typ = 'frame',line = False)

​		1）将JSON格式准换成默认的Pandas DataFrame格式

​		2）orient：string,Indication of expected JSON string format

​			'split'：dict like {index -> [index],columns -> [columns],data -> [values]}

​			'records'：list like [{columns ->value},.....,{columns -> value}]

​			'index'：dict like {index -> {columns -> value}}

​			'columns'：dict like {column -> {index -> value}},默认格式

​			'values'：just the values array

​		3）lines：boolean,default False

​			按照每行读取json对象

​		4）typ：default 'frame'，指定转换成的对象类型series或者dataframe

``` python
sa = pd.read_json("文件路径",orient = "records",lines = True)
```

​	（2）DataFrame.to_json()

``` python
sa.to_json("要保存到的位置",orinet = "records",lines = True) # 有必要设置lines
```

