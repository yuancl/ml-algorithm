训练步骤：
1.实现正则化的cost function和gradient，梯度公式的推导可以参考exe1
2.查看数据的大致分布
3.实现Learning curve,展示的是training error和cv error随数据集的变化的曲线，从这个曲线可以观察出bias，variance等
实现的方式是对数据集进行切分，数据集逐渐变大，然后对切分后的数据集分布进行训练，分别求出training error和cv error
4.poly_features,feature_normalize实现
5.用多项式去刻画数据集，同时正则化参数lambda设置为0
6.设置正则化参数lambda为不同的值，训练全部数据集，看loss error随lambda值的变化情况，找到最合适的lambda值
7.使用确定的lambda值，再去训练training数据集，最后进行test数据集的评测



经验：
一.对传入的X，尽量统一为都是天级了X0项的向量
二.Learning curve:
1.train error和cv error曲线都收敛，但是error值都比较高的话，一般是bias情况，同时也可以对比看h(x)对数据的可以刻画情况
2.train error值很小，在0附件，但是cv error收敛后值比较大，也就是train 和 cv存在gaps(基本不会重合)，这种情况一般都是high virance情况。这个时候的h(x)
一般能精确的刻画数据
3.train error和cv error曲线都收敛，并且收敛后值error都比较小，这种情况是比较合理的
三.bias情况可以使用多项式去拟合，减小error
四.标准化，如果不标准化，那么有些数值会变的非常大，所有标准化也能加速收敛
五.通过training error,cv error随lambda的变化曲线图可以找到比较合适的lambda值
六.获取到lambda值，进行training data的训练，得到模型h(x)，然后用此模型进行test dataset的测试评测，主要test dataset也需要进行多项式话和
标准化的