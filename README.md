# Probabilistic Matrix Factorization

     Salakhutdinov R, Mnih A. Probabilistic Matrix Factorization[C]// International Conference on Neural Information Processing Systems. 2007.
     
     很多现有的协同过滤的方法既不能处理非常大的数据集，也不能容易地应对有非常少的评价的用户。在这篇论文中，作者提出了概率矩阵分解（PMF）模型，它的规模与观察值的数目线性相关，并且更重要的是，它在非常大的、稀疏的和非常失衡的Netflix数据集上表现优异。
    
     文章更进一步地扩展PMF模型在模型参数中加入一个合适的先验并且展示模型能力怎样可以被自动地控制。
    
     最后，文章引入一个有约束版本的PMF模型，它基于评价相似集合电影的用户可能有相似偏好的假设。这个结果模型对于有很少评价的用户能够归纳的相当好。
    
     当多个PMF模型的预测与受限玻尔兹曼机模型的预测线性结合时，文章达到了0.8861的误差率，这几乎比Netflix自己系统的分数提升了7%。 

# 数据集
## Netflix数据集
这个数据集来自于电影租赁网址Netflix的数据库。Netflix于2005年底公布此数据集并设立百万美元的奖金(netflix prize)，征集能够使其推荐系统性能上升10％的推荐算法和架构。这个数据集包含了480189个匿名用户对大约17770部电影作的大约10亿次评分。

文章使用数据集为Netflix数据集，由于算力资源限制，本项目选择MovieLens数据集作为实验数据集。
## MovieLens 数据集
MovieLens数据集包含多个用户对多部电影的评级数据，也包括电影元数据信息和用户属性信息。

这个数据集经常用来做推荐系统，机器学习算法的测试数据集。尤其在推荐系统领域，很多著名论文都是基于这个数据集的。

(PS: 它是某次具有历史意义的推荐系统竞赛所用的数据集)。

### MovieLens 数据集下载地址
http://files.grouplens.org/datasets/movielens/

该数据有好几种版本，对应不同数据量，本项目所用的数据为1M的数据，包含6040个独立用户对3900部电影作的大约100万次评分。


     1M： 6,040个用户对于3,900部电影的1,000,209个评分

     时间：2000年

     包含ratings.dat、users.dat、movies.dat

     Ratings.dat: 用户id、电影id、评分（1~5）、时间标签

     Users.dat： 性别、年龄、职位、邮编

     Movies.dat: 电影id、标题、流派
     
##本项目数据集
本项目数据集基于MovieLens 数据集1M版本数据，对Ratings.dat做了处理，得到了程序所需要的rating_table.csv

# cpmf.py

该文件是文章中Constrained Probabilistic Matrix Factorization 模型的实现代码。

运行主函数即可读入数据、训练模型、保存模型、执行预测。

# rating_table.csv

预处理后的MovieLens 数据集1M版本数据，包含6,040个用户对于3,900部电影的1,000,209个评分。

格式为：

    [user_id, movie_id, rating]
    

# U.csv && V.csv

U.csv && V.csv 分别为训练得到的用户特征矩阵和电影特征矩阵。

# [remote rejected] master -> master (pre-receive hook declined)
总是push rejected，不知道为什么