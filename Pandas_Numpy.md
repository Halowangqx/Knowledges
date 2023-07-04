# Pandas



## DataFrame

> 本质上是一个多维字典

基本信息

```python
data.index

data.columns
# 如果是单层索引，就会返回一个Index类 列表，每个元素是列标签
# 如果是多层索引，就会返回MultiIndex类， 列表，每个元素是一个tuple

data.values
# 返回内置的数组结构，是numpy.ndarray
```

dataframe 的生成

```python 
pd.DataFrame()
（二维ndarray，index= ， column= ）

（由list/array/tuple组成的字典）# 这实际上是缺省一维的，字典的键值会作为列

（由Series/字典组成的字典） # 这个不缺省！
```

## compute & joint

**+**

- 数乘就是逐元素的倍数

- 两个dataframe相加
  可以对齐则对位相加，否则用NaN填充

- 如果参与运算的一个是DataFrame，另一个是[Series](https://so.csdn.net/so/search?q=Series&spm=1001.2101.3001.7020)，那么pandas会对Series进行行方向的[广播](http://liao.cpython.org/numpy08/#84)，然后做相应的运算。

### **concat**

```python
pd.concat(a,b,axis= , keys=['from_a', 'from_b'], names= ['index1','index2'])
#函数可以        沿着指定的轴将多个dataframe或者series拼接到一起

###### params:

keys 
# 为原组数据建立自己的多层索引 
# 可以用来标示数据的来源
names
# 往往和keys结合使用， 当有MultiIndex时
# 为index/column（由axis指定的）设置名字
join
# 相当于merge和join中的how
# 可以选择 inner（交） 或者outer（并）
axis
# axis=0 就是保留所有共行的信息
# axis=1 就是保留所有共有列信息
```

### **merge**

pandas.merge(_left_, _right_, _how='inner'_, _on=None_, _left_on=None_, _right_on=None_, _left_index=False_, _right_index=False_, _sort=False_, _suffixes=('_x', '_y')_, _copy=True_, _indicator=False_, _validate=None_)
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html

本质上是将两个dataframe按照一个column或者index进行对齐合并

> `how`指定了合并的方式

> 其中`on\right_on\left_on\left_index\right_index`这几个参数都是为了指定按照什么进行对齐

> 当对齐的行/列中出现重复的时候：**内积的感觉**
>
> - one-to-one join 
> - many-to-one join: 单值记录会转化成多个![[Pasted image 20230320162430.png]]
>
> - many-too-many join:  ![[Pasted image 20230320162654.png]]



**一般都用merge， join甚至不能rigt on**



### **join**

DataFrame.join(other, *on=None***,** *how='left'***,** *lsuffix=''***,** *rsuffix=''***,** *sort=False***,** *validate=None***)

```python
**other** DataFrame, Series, or a list containing any combination of them

Index should be similar to one of the columns in this one. If a Series is passed, its name attribute must be set, and that will be used as the column name in the resulting joined DataFrame.

**on**     str, list of str, or array-like, optional

按照哪个列来进行融合

Column or i**ndex level name(s) in the caller** to join on the **index in other**, otherwise joins index-on-index.



**how** {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘left’

How to handle the operation of the two objects.

- left: use calling frame’s index (or column if on is specified)
- right: use other’s index.
- outer: form union of calling frame’s index (or column if on is specified) with other’s index, and sort it. lexicographically.
- inner: form intersection of calling frame’s index (or column if on is specified) with other’s index, preserving the order of the calling’s one.
- cross: creates the cartesian product from both frames, preserves the order of the left keys.
```





## group

```python
DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=NoDefault.no_default, squeeze=NoDefault.no_default, observed=False, dropna=True)

by
# Used to determine the groups for the groupby

axis
# Split along rows (0) or columns (1)

level
# If the axis is a MultiIndex (hierarchical), group by a particular level or levels. Do not specify both `by` and `level`.

as_index
# For aggregated output, return object with group labels as the index. Only relevant for DataFrame input. as_index=False is effectively “SQL-style” grouped output.

```

**Attention:**

- 在groupby以後，生成的是dataframe的集合，by=不會被捨棄，而是會繼續作爲column存在

- 可以用[]來在所有dataframe中指定所需要的列 然後使用apply，這時返回的是一個Series。可以使用`.to_frame()`轉化成dataframe,用`.rename(column=<dict>)`更改列名

  此時的行索引應該是by=的分組

- 返回的對象是groupby對象。這是一種中間的數據結構，不可以直接索引或遍歷

  可以使用 `get_group()` 方法获取某个子组的数据。



## apply/map/agg

### apply/map

> Apply a function along an axis of the DataFrame.

> Objects passed to the function are Series objects whose index is either the DataFrame’s index (`axis=0`) or the DataFrame’s columns (`axis=1`). By default (`result_type=None`), the final return type is inferred from the return type of the applied function. Otherwise, it depends on the result_type argument.

- **axis**{0 or ‘index’, 1 or ‘columns’}, default 0
  Axis along which the function is applied:
- 0 or ‘index’: apply function to each column.
- 1 or ‘columns’: apply function to each row.

```python
DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)

raw
# Determines if row or column is passed as a Series or ndarray
```

- result_type{‘expand’, ‘reduce’, ‘broadcast’, None}, default None

These only act when `axis=1` (columns):

-   ‘expand’ : list-like results will be turned into columns.
-   ‘reduce’ : returns a Series if possible rather than expanding list-like results. This is the opposite of ‘expand’.
-   ‘broadcast’ : results will be broadcast to the original shape of the DataFrame, the original index and columns will be retained.



**apply可作用於一個dataframe或者一個Series**

- dataframe時，傳入參數函數的是Series（類）
- series時，傳入參數函數的是item



**map只能作用於Series**

`map()` 函数可以接受一个字典、Series 或函数，并将其映射到 Series 中的每个元素上。如果传入的是字典或 Series，则会将它们中的键值对应；如果传入的是函数，则会将函数应用于每个元素。因此，`map()` 通常用于对一列数据进行操作。



### agg

#### for dataframe

```python
### use list 一視同仁
>>> df.agg(['sum', 'min'])
        A     B     C
sum  12.0  15.0  18.0
min   1.0   2.0   3.0

### use dict 向上一層 可以指定作用的元素了
### {value:list}
>>> df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})
        A    B
sum  12.0  NaN
min   1.0  2.0
max   NaN  8.0

### use tuple 再向上一層 可以指定index
### index = (value,list)
>>> df.agg(x=('A', max), y=('B', 'min'), z=('C', np.mean))
     A    B    C
x  7.0  NaN  NaN
y  NaN  2.0  NaN
z  NaN  NaN  6.0
```

#### for groupby object



## Sort

DataFrame.sort_values(_by_, _*_, _axis=0_, _ascending=True_, _inplace=False_, _kind='quicksort'_, _na_position='last'_, _ignore_index=False_, _key=None_)

> Sort by the values along either axis.
> https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html?highlight=sort_values#pandas.DataFrame.sort_values



## index

- 下标索引
  https://blog.csdn.net/p1306252/article/details/114879951

```python
# 列索引
data.column_name
data['column_name']
# 混合索引  维度之间用','分割
data.loc[row_name,column_name]
data.iloc[row_num, column_num]

##### 通过[]i.e.list 引用多个索引
##### 通过()i.e.tuple 引用不同维度上的索引
# df.loc[('A', slice(None), 'Z'), :]
```

- slice对象

  slice（None)表示所有

  slice(A,B,step) 注意是闭区间

  

- **setindex** 改变column的名字

```python
# 将一个column设置为index
dataframe.setindex(column_name, inplace= , drop= )

# 也可以直接设置索引
>>> better_index = ["X1", "X2", "Y1", "Y2", "Y3"]
>>> df0.index = better_index
```

- **reset_index()** 将索引转化成自然数

```python
DataFrame.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill=”)
# 将索引转化成自然数 （同时可选择将原index加入column）

# **level:**int，字符串或列表以选择并从索引中删除传递的列。  
# **drop:**布尔值，如果为False，则将替换的索引列添加到数据中。
```

- **reindex()** 

DataFrame.reindex(_labels=None_, _index=None_, _columns=None_, _axis=None_, _method=None_, _copy=None_, _level=None_, _fill_value=nan_, _limit=None_, _tolerance=None_)

> Conform Series/DataFrame to new index with optional filling logic.
>  Places NA/NaN in locations having no value in the previous index. A new object is produced unless the new index is equivalent to the current one and `copy=False`.

- 按照条件筛选数据

```python
stripes_or_bars=data[(data['stripes']>=1) | (data['bars']>=1)]
# 按照布尔值筛选 里面也可以是返回布尔值的函数
```



**Attention!!!**

> The index of X is 'movieid'  and the index of P is just number
>
> Why I can's use `P.iloc[:,movie][X.iloc[user,:] != 0]`  where P X are both pd.dataframe? 

The issue in your expression `P.iloc[:, movie][X.iloc[user, :] != 0]` is that you are trying to **use boolean indexing on a pandas Series with a boolean Series that has a different index**.



## multi-index

The [`MultiIndex`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html#pandas.MultiIndex "pandas.MultiIndex") object is the hierarchical analogue of the standard [`Index`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.html#pandas.Index "pandas.Index") object which typically stores the axis labels in pandas objects. You can think of `MultiIndex` as an array of tuples where each tuple is unique. A `MultiIndex` can be created from a list of arrays (using[`MultiIndex.from_arrays()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.from_arrays.html#pandas.MultiIndex.from_arrays "pandas.MultiIndex.from_arrays")), an array of tuples (using [`MultiIndex.from_tuples()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.from_tuples.html#pandas.MultiIndex.from_tuples "pandas.MultiIndex.from_tuples")), a crossed set of iterables (using [`MultiIndex.from_product()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.from_product.html#pandas.MultiIndex.from_product "pandas.MultiIndex.from_product")), or a [`DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame "pandas.DataFrame") (using[`MultiIndex.from_frame()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.from_frame.html#pandas.MultiIndex.from_frame "pandas.MultiIndex.from_frame")). The `Index` constructor will attempt to return a `MultiIndex` when it is passed a list of tuples




### 索引

`data.iloc[:3,:3]` 可以直接按照二维表切片
`data.loc[]` 可以实现最外层的切片

**Use pd.indexSlice**
',' let us from row to column
https://blog.csdn.net/weixin_47911946/article/details/118003908

> slice in python
> https://www.w3schools.com/python/ref_func_slice.asp



### **unstack()**

It essentially moves one of the index levels to the column axis, creating a new DataFrame with multi-level columns. 

**index2column**

### swaplevel()/sort_index

```python
datafame.swaplevel(i, j, axis=)
```

swap是单纯将tuple反过来了，并没有按照新的高层index/column排序，所以需要重新排序

```python
sort_index(axis=，level=, ascending=, inplace=)
```









## drop

```python
df0.drop_duplicates("team", ignore_index=True)
```


## sample

```python
# 采样
DataFrame.sample(self: ~ FrameOrSeries, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None) 

n  frac
# 采样条数 采样比例 选一个设置 frac可以上采样

axis
# 

random_state
# seed
```



## Fuctions

- pd.pivot_table()
- dataframe.descibe





# Numpy

## Data Operation

### Broadcast

> When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing (i.e. rightmost) dimension and works its way left. Two dimensions are compatible when
>
> 1. they are equal, or
> 2. one of them is 1.（Note that missing dimensions are assumed to have size one.)  ---> Broadcasting

![A 1-d array with shape (3) is stretched to match the 2-d array of shape (4, 3) it is being added to, and the result is a 2-d array of shape (4, 3).](https://numpy.org/doc/stable/_images/broadcasting_2.png)

![A huge cross over the 2-d array of shape (4, 3) and the 1-d array of shape (4) shows that they can not be broadcast due to mismatch of shapes and thus produce no result.](https://numpy.org/doc/stable/_images/broadcasting_3.png)

​	When the **trailing dimensions** of the arrays are unequal. broadcasting fails.

### Operators

**high-dimention** can be constidered as **batch**

In all the following method, the extra dimentions will be seen **as batch**!

- np.matmul(a, b)    @

  matrix multiplication

  - *for two one-dim arrays, it perform inner product*

- *

  multiply the matrices elementwise

- np.dot

  总的来说，就是可以做矩阵乘法	就做矩阵乘法，可以做内积，就做内积

  - for two 0-dim arrays (scalars), just multiply 
  - *for two one-dim arrays, it perform inner product * 内积

  - for two two-dims arrays, it perform matrix multiplication 

  - If *a* is an N-D array and *b* is a 1-D array, it is a sum product over the last axis of *a* and *b*.

  - If *a* is an N-D array and *b* is an M-D array (where `M>=2`), it is a sum product over the last axis of *a* and the second-to-last axis of *b*:

    ```python
    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    ```

    (有点像矩阵乘法)

- np.inner(a, b)

  Ordinary inner product of vectors for 1-D arrays (without complex conjugation), in higher dimensions a sum product over the last axes.

- `no.vdot() `will compute the inner product, **flatten the N-d**.

### np.ufunc.

NumPy's universal function (ufunc) **class**, which operates on ndarrays element-wise. 

They are used to provide ***vectorization*** in numpy.

ufuncs also take additional arguments, like:

- `where` boolean array or condition defining where the operations should take place.

- `dtype` defining the return type of elements.

- `out` output array where the return value should be copied.

**Examples：**

1. Arithmetic operations: `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.power` 事实上，numpy中的运算符号+, - , *, / etc.都是这些ufunc functions.
2. Trigonometric functions: `np.sin`, `np.cos`, `np.tan`
3. Exponential and logarithmic functions: `np.exp`, `np.log`, `np.log10`
4. Comparison operations: `np.greater`, `np.less`, `np.equal`
5. Bitwise operations: `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`



下面是ufunc class 中的一些methods:

```python
np.ufunc.reduce

np.ufunc.accumulate

np.ufunc.outer

# conduct the outer operation, and each pair of enties applied  the "ufunc".

np.ufunc.at

# Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’. 
a = np.array([1, 2, 3, 4])
>>> np.add.at(a, [0, 1, 2, 2], 1)
>>> a
array([2, 3, 5, 4])
```



## index

- 正常的索引其实是[()]其中是一个**tuple**。每个维度是tuple的一项，所以之间采用'**,**'分割

- **Advanced indexing**

  > Advanced indexing is triggered when the selection object, **obj**, is a non-tuple sequence object, an [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) (of data type integer or bool), or a tuple with at least one sequence object or ndarray (of data type integer or bool). There are two types of advanced indexing: integer and Boolean.

  - **Integer array indexing**

    > Integer array indexing allows selection of arbitrary items in the array based on their *N*-dimensional index. Each integer array represents a number of indices into that dimension.

    

  - **Boolean array indexing**

    > If `obj.ndim == x.ndim`, `x[obj]` returns a 1-dimensional array filled with the elements of *x* corresponding to the [`True`](https://docs.python.org/3/library/constants.html#True) values of *obj*. The search order will be [row-major](https://numpy.org/doc/stable/glossary.html#term-row-major), C-style. If *obj* has [`True`](https://docs.python.org/3/library/constants.html#True) values at entries that are outside of the bounds of *x*, then an index error will be raised. If *obj* is smaller than *x* it is identical to filling it with [`False`](https://docs.python.org/3/library/constants.html#False).

    the obj has (or after broadcast has) **the same shape** as the x, which **filled with True and False**.

    ```python
    # select entries from an array which is not Nan
    x[~np.isnan(x)]
    # select entries from an array which is less than 20
    # the scalar "20" will be broadcast
    x[x < 20]
    ```

    > **A very useful way to understand!**
    >
    > In general if an index includes a Boolean array, the result will be identical to inserting `obj.nonzero()` into the same position and using the integer array indexing mechanism described above. `x[ind_1, boolean_array, ind_2]` is equivalent to `x[(ind_1,) + boolean_array.nonzero() + (ind_2,)]`.

  ​	

## Copies and views

> The NumPy array is a data structure consisting of two parts: the [contiguous](https://numpy.org/doc/stable/glossary.html#term-contiguous) data buffer with the actual data elements and the metadata that contains information about the data buffer. The metadata includes data type, strides, and other important information that helps manipulate the [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) easily.

- **view**: access the array differently by just changing certain metadata like sride and dtype. ----> create new arrays called view

  *the data buffer remains the same*

- **copy**: crate by duplicating the databuffer as well as the metadata.

  *brand new data buffer*

  

- basic indexing always creates views.

- [Advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing), on the other hand, always creates copies



- The [`base`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.base.html#numpy.ndarray.base) attribute of the ndarray makes it easy to tell if an array is a view or a copy. The base attribute of a view returns the original array while it returns `None` for a copy.






**the elements stored in a NumPy ndarray object need to be of the same type.**

If you try to create an ndarray with elements of different types, NumPy will automatically convert the elements to a common type



## Initialize ndarrays

```python
import numpy as np

##### all of them has param: dtype
np.zeros()
np.empty()
np.ones()
np.ones_like()
np.zeros_like()

# creat a constant array
np.full(shape, fill_value， dtype, order)
np.full_like()

# Creat an identity matrix
np.eye(N, M, k)
# k: Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.

# Create an array filled wieth random numbers [0.0, 1.0]
np.random.random(shape)



```

##　Aggregation Function

sum mean prod std var min max argmin argmax etc.



**axis**

When we say "apply/conduct the function in the axis i", 就是按照非i的原则分组。

将所有     *维度i下标不同*    *其余维度下标相同*    的元素分成一组，然后在这个组中计算

```python
x = np.arange(2*2*2).reshape(2,2,2)
array([[[0, 1],
        [2, 3]],

       [[4, 5],
        [6, 7]]])

x.mean(0)
array([[2., 3.],
       [4., 5.]])

x.mean((0, 1))
array([3., 4.])

x.mean(1)
array([[1., 2.],
       [5., 6.]])


```



## some useful methold

### where

**numpy.****where****(***condition***,** **[***x***,** *y***,** **]***/***)**

Return elements chosen from *x* or *y* depending on *condition*.



condition“可以是elementwise"d的 

比如ndarray<ndarray 这样产生的就是一个逐元素大小比较值的 bool ndarray

```python
a = np.array([[0, 1, 2],
              [0, 2, 4],
              [0, 3, 6]])
# a<4 其实就是a中逐个元素和4比较
# step 1: 4广播成a的形状
# step 2: 两个ndarray用'<'相连 会逐元素处理 
# step 3: numpy会返回一个与原数组a大小相同的bool类型的数组
np.where(a < 4, a, -1)  # -1 is broadcast
array([[ 0,  1,  2],
       [ 0,  2, -1],
       [ 0,  3, -1]])
```



### np.repeat / np.tile

**np.tile(A, reps)**

> Construct an array by repeating A the number of times given by reps.
>
> If *reps* has length `d`, the result will have dimension of `max(d, A.ndim)`.
>
> If `A.ndim < d`, *A* is promoted to be d-dimensional by prepending new axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not the desired behavior, promote *A* to d-dimensions manually before calling this function.
>
> If `A.ndim > d`, *reps* is promoted to *A*.ndim by pre-pending 1’s to it. Thus for an *A* of shape (2, 3, 4, 5), a *reps* of (2, 2) is treated as (1, 1, 2, 2).\

**Example:**

```python
a:
[[1 2]
 [3 4]]

#####: repeat 是elementwise的复制，每个元素在原位复制3遍，
np.repeat(a, 3, axis=0):
[[1 2]
 [1 2]
 [1 2]
 [3 4]
 [3 4]
 [3 4]]

np.repeat(a, 3, axis=1)
[[1 1 1 2 2 2]
 [3 3 3 4 4 4]]

##### tile 在rep中应该指定所有维度 否则会自动从“从后向前”匹配
# tile 的复制是对一个维度整体复制
np.tile(a, 3) ----> np.tile(a, (1, 3))
[[1 2 1 2 1 2]
 [3 4 3 4 3 4]]
```



### np.bincount

**np.bincount(x, weights=None, minlength=0)**

> The number of bins (of size 1) is one larger than the largest value in *x*. If *minlength* is specified, there will be at least this number of bins in the output array (though it will be longer if necessary, depending on the contents of *x*). Each bin gives the number of occurrences of its index value in *x*. If *weights* is specified the input array is weighted by it, i.e. if a value `n` is found at position `i`, `out[n] += weight[i]` instead of `out[n] += 1`.



这个函数用于聚类的例子：

```python 
# recompute centroids as the mean of the data points in each cluster
# clusters中值为i的有centroids_num[i]个 ---> “桶”聚类
centroids_num = np.bincount(clusters, minlength=K)
#　相当于　按照某种条件求和
centroids_x = np.bincount(clusters, weights=X[:, 0], minlength=K)
centroids_y = np.bincount(clusters, weights=X[:, 1], minlength=K)
# 计算了“桶”聚类后各个元素不同维度上的平均值
centroids[:, 0] = centroids_x / centroids_num
centroids[:, 1] = centroids_y / centroids_num
```





## examples

how to inplement a convolution funtion

````python
import numpy as np
def func_1(img, kernel):
    # inplement a convolution function
    
    # get the size of the img and the kernal
    h, w = img.shape
    hk, wk = kernel.shape
    
    i0,j0 = np.meshgrid(range(hk),range(wk),indexing='ij')
    i1,j1 = np.meshgrid(range(h-hk+1),range(w-wk+1),indexing='ij')
    i = i0.reshape(-1,1)+i1.reshape(1,-1)
    j = j0.reshape(-1,1)+j1.reshape(1,-1)
    select_img = img[i,j] # (hk*wk, h*w)
    weights = kernel.reshape(1,-1) # (1,hk*wk)
    output = weights @ select_img
    output = output.reshape(h-hk+1,w-wk+1)
    return output

````











## 数据读入

- `tofile()`函数用于将NumPy数组写入磁盘。该函数可以将数组以二进制格式写入磁盘，其中数组的每个元素都按顺序存储。如果要将数组以ASCII文本格式写入磁盘，可以指定参数`sep`，例如`sep=','`表示用逗号分隔元素。
- `fromfile()`函数用于从磁盘读取NumPy数组。该函数可以将以二进制格式存储的数组读入内存，返回一个NumPy数组。要指定要读取的数组的形状和数据类型，可以使用参数`dtype`和`shape`。













## impressive 

### elementwise arithmatic manipulation





```python
# calculate the amount of positive number in the img	
np.sum(np.where(img > 0, 1, 0))

```





