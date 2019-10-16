import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers


class BiLSTM:

    def __init__(self, lstm_dim, num_tag, lr):

        self.lstm_dim = lstm_dim  # 隐含层维度
        self.num_tag = num_tag  # tag数：B M E S
        self.learning_rate = lr  # 学习率
        self.initializer = initializers.xavier_initializer()
        # 初始化方法使用xavier Initialization，以使网络中的信息更好地流动，网络各层激活值及状态梯度的方差应尽量保持不变
        
