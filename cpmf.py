import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series


class BPMF():
    def __init__(self, num_feature=30, learning_rate=0.05, momentum=0.9, regu_y=0.002, regu_v=0.002, regu_w=0.002,
                 K=5.0):

        self.num_feature = num_feature  # 潜在特征向量维数
        self.learning_rate = learning_rate  # 学习率
        self.momentun = momentum  # 动量
        self.regu_y = regu_y  # 用户特征正则系数
        self.regu_v = regu_v  # 电影特征正则系数
        self.regu_w = regu_w
        self.K = K  # 评分区间最大值

    def train(self, num_user, num_mv, user_list, mv_list, data_train, data_test, max_iter=100):
        '''
        训练函数，输入划分好的数据集和参数，输出所需的特征矩阵
        :param num_user: 用户数
        :param num_mv: 物品数
        :param user_list: 用户ID列表
        :param mv_list: 物品ID列表
        :param data_train: 训练数据集
        :param data_test: 测试数据集
        :param max_iter: 最大迭代次数
        :return:
        '''

        # 初始化特征矩阵，假设特征矩阵服从高斯分布
        U = DataFrame(np.random.normal(0, 1, [self.num_feature, num_user]), columns=user_list)
        Y = DataFrame(np.random.normal(0, 1, [self.num_feature, num_user]), columns=user_list)
        V = DataFrame(np.random.normal(0, 1, [self.num_feature, num_mv]), columns=mv_list)
        W = DataFrame(np.random.normal(0, 1, [self.num_feature, num_mv]), columns=mv_list)

        # 设置早停法阈值
        pre_rmse = 100.0  # 上次迭代后的rmse，初始化为一个较大的数值
        endure_count = 3  # 容忍次数
        patience = 0  # 用于记录rmse连续不下降的次数

        # 开始迭代训练
        for iter in range(max_iter):
            loss = 0.0
            for data in data_train:
                u_id = data[0]
                m_id = data[1]
                rating = self.rating_map(data[2])  # 映射到（0,1）区间后的评分

                # 通过logistic函数后的预测值,取值在（0,1）之间
                pre_rating = self.logistic(np.dot((Y[u_id] + W[m_id]).T, V[m_id]))
                error = rating - pre_rating
                loss += error ** 2  # 经验误差

                # 随机梯度下降更新特征矩阵，由目标函数求偏导得到
                U[u_id] += self.learning_rate * (
                        error * pre_rating * (1 - pre_rating) * V[m_id] - self.regu_y * Y[u_id])
                V[m_id] += self.learning_rate * (
                        error * pre_rating * (1 - pre_rating) * (Y[u_id] + W[m_id]) - self.regu_v * V[m_id])
                W[m_id] += self.learning_rate * (
                        error * pre_rating * (1 - pre_rating) * V[m_id] - self.regu_w * W[m_id])
                # 将正则误差加入loss
                loss += self.regu_y * np.square(Y[u_id]).sum() + self.regu_v * np.square(
                    V[m_id]).sum() + self.regu_w * np.square(W[m_id]).sum()
            loss = 0.5 * loss

            rmse = self.rmse(U, V, data_test)
            print('Iter:%d    loss:%.5f    rmse:%.5f' % (iter + 1, loss, rmse))

            # 早停法：若测试rmse连续3次不再下降（或者上升），可视为收敛，停止训练
            if rmse < pre_rmse:
                pre_rmse = rmse
                patience = 0
            else:
                patience += 1
            # 连续不下降次数超过3次停止训练
            if patience >= endure_count:
                break

        # 提前停止训练或者达到最大迭代轮数，保存模型到文件
        self.save_model('.', U, V)
        pass

    def save_model(self, path, U, V):
        '''
        模型保存函数，将训练得到的特征矩阵保存至指定路径
        :param path: 模型保存路径
        :param U: 用户特征矩阵
        :param V: 物品特征矩阵
        :return:
        '''
        U.to_csv(path + '/U.csv')
        V.to_csv(path + '/V.csv')
        print('Model has been save.')
        pass

    def load_model(self, path):
        '''
        模型加载函数，读取特征矩阵文件，返回预测需要的特征矩阵格式
        :param path: 特征矩阵文件路径
        :return:
        '''
        table_u = pd.read_csv(path + '/U.csv')
        table_v = pd.read_csv(path + '/V.csv')
        table_u.drop(['Unnamed: 0'], axis=1, inplace=True)
        table_v.drop(['Unnamed: 0'], axis=1, inplace=True)
        U_value = table_u.values
        V_value = table_v.values
        return U_value, V_value
        pass

    def predict(self, path, U, V):
        '''
        评分预测函数，根据用户和物品特征矩阵进行评分预测
        :param path: 评分预测文件写入路径
        :param U: 用户特征矩阵
        :param V: 物品特征矩阵
        :return:
        '''

        pre_matrix = self.logistic(U.T.dot(V))  # 当前预测在区间（0,1）上
        # 将（0,1）的预测值重新映射回区间（1,5）
        rating_matrix = pre_matrix * (self.K - 1) + 1

        # 将生成的预测矩阵写入文件保存
        writer = DataFrame(rating_matrix)
        writer.to_csv(path + '/pre_rating.csv')
        print('Prediction has been made.')
        pass

    def rmse(self, U, V, data_test):
        '''
        计算均方根误差
        :param U: 用户特征矩阵
        :param V: 物品特征矩阵
        :param data_test: 验证数据集
        :return:
        '''

        num_items = 0
        tmp_rmse = 0.0
        for item in data_test:
            num_items += 1
            u_id = item[0]
            m_id = item[1]
            rating = self.rating_map(item[2])
            pre_rating = self.logistic(np.dot(U[u_id].T, V[m_id]))
            tmp_rmse += np.square(rating - pre_rating)
        rmse = np.sqrt(tmp_rmse / num_items)
        return rmse

    def logistic(self, prediction):
        '''
        logistic函数，将预测评分映射至（0,1）区间
        :param prediction: 预测评分
        :return:
        '''
        power = (-1) * prediction
        return 1 / (1 + np.exp(power))

    def rating_map(self, rating):
        '''
        已有评分映射函数，将评分矩阵观测评分映射至（0,1）区间
        :param rating: 观测评分
        :return:
        '''
        return (rating - 1) / (self.K - 1)


def data_read(path, ratio):
    '''
    数据读入，划分训练集和验证集
    :param path: 数据文件路径
    :param ratio: 训练集占所有数据比例
    :return:
    '''
    table = pd.read_csv(path)
    # table.drop(['Unnamed: 0'], axis=1, inplace=True)
    value_array = table.values

    user_set = {}
    num_user = 0
    mv_set = {}
    num_mv = 0
    for line in value_array:
        user_id, mv_id, rating = line
        if user_id not in user_set:
            user_set[user_id] = user_id
            num_user += 1
        if mv_id not in mv_set:
            mv_set[mv_id] = mv_id
            num_mv += 1
    user_list = user_set.keys()
    mv_list = mv_set.keys()

    np.random.shuffle(value_array)
    """
    small_set=value_array[:int(len(value_array)/100)]
    data_train = small_set[:int(len(small_set) * ratio)]
    data_test = small_set[int(len(small_set) * ratio):]
    """
    data_train = value_array[:int(len(value_array) * ratio)]
    data_test = value_array[int(len(value_array) * ratio):]

    print('Data prepare has been done.\n',
          'The data set includes %d users, %d movies and %d ratings in total.' % (num_user, num_mv, len(value_array)))
    return num_user, num_mv, user_list, mv_list, data_train, data_test


if __name__ == '__main__':
    n_user, n_mv, user_list, mv_list, data_train, data_test = data_read('./rating_table.csv', 0.8)
    pmf = BPMF()
    # pmf.train(n_user, n_mv, user_list, mv_list, data_train, data_test)    # 首次训练需要取消本行注释以生成特征矩阵文件U和V
    U, V = pmf.load_model('.')
    pmf.predict('.', U, V)
