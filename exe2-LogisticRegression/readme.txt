总结：
1.学习coursera上第三周的课程
2.阅读ext2.pdf
3.编程非正则化的cost_fun和gradient，然后使用优化函数minimize进行多轮迭代优化(这里就不用像线下回归自己定义alpha，迭代次数进行优化了),对data1数据进行拟合，为线性方程就可以拟合了
4.查看了data2的数据分布，线性方程不能很好的拟合，就需要使用高阶多项式进行拟合(PolynomialFeatures来构造特征)，这种容易出现过拟合现象(最后设置lamba为0就出现了过拟合),所以就用正则化的优化方法处理
5.使用正则化方法是，区分L1,L2正则方法，同时注意对cost和gradient都使用正则化:cost_fun_reg, gradient_reg,同时注意theta从1开始，最后同样使用minimize对正则化后的方法进行迭代优化
6.最后设置不用lambda的值，看拟合情况：lambda过大(100)出现bias，设置过小(0)出现varance


重点：理解逻辑回归算法(sigmoid,cost_fun等)，然后理解高阶多项式带来的问题，已经使用正则化如何处理


心得：
1.一维数组与二维数组的乘积与加减：乘积的时候，在左边可作为1*n二维数组处理：a.dot(xxx)。在右边可以作为n*1二维数组处理：xxx.dot(a)
当加减法的时候，最好和1*n的二维数组相加减
2.在算法上，经常对theta或者是一些方法的返回值用作一维数组
3.python中，参数为一个数组时，会对数组中的所以值进行处理，比如sigmoid方法，可以传如二维数组
4.一维数组操作a[xx:xx].二维数组操作a[xx,xx],a[0:2,xx]二维数组中间的操作又可按照一维数组操作
