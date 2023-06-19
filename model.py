import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore.train import Model
from mindspore.common.initializer import initializer, XavierNormal, Zero

class SubNet(nn.Cell):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        :param in_size: input dimension
        :param hidden_size: hidden layer dimension
        :param dropout:  dropout probability
        :return (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(keep_prob=1-dropout)
        self.linear_1 = nn.Dense(in_size, hidden_size)
        self.linear_2 = nn.Dense(hidden_size, hidden_size)
        self.linear_3 = nn.Dense(hidden_size, hidden_size)

    def construct(self, x):
        '''
        :param x: tensor of shape (batch_size, in_size)
        :return: tensor of shape (batch_size, hidden_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = ops.relu(self.linear_1(dropped))
        y_2 = ops.relu(self.linear_2(y_1))
        y_3 = ops.relu(self.linear_3(y_2))

        return y_3

class TextSubNet(nn.Cell):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        :param in_size: input dimension
        :param hidden_size: hidden layer dimension
        :param out_size:
        :param num_layers: specify the number of layers of LSTMs
        :param dropout: dropout probability
        :param bidirectional: specify usage of bidirectional LSTM
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(keep_prob=1-dropout)
        self.linear_1 = nn.Dense(hidden_size, out_size)

    def construct(self, x):
        '''
        :param x: tensor of shape (batch_size, sequence_len, in_size)
        :return: tensor of shape (batch_size, sequence_len, out_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

class LMF(nn.Cell):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, text_out, dropouts, output_dim, rank, use_softmax=False):
        '''
        :param input_dims: a length-3 tuple, contains (audio_dim, video_dim, text_dim)
        :param hidden_dims:  another length-3 tuple, hidden dims of the sub-networks
        :param text_out: int, specifying the resulting dimensions of the text subnetwork
        :param dropouts: a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
        :param output_dim: int, specifying the size of output
        :param rank: int, specifying the size of rank in LMF
        :param use_softmax:
        :return (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        #dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.text_out = text_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        #define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(keep_prob=1-self.post_fusion_prob)
        self.audio_factor = ms.Parameter(ms.Tensor(shape=(self.rank, self.audio_hidden + 1, self.output_dim), dtype=ms.float32, init=XavierNormal()))
        self.video_factor = ms.Parameter(ms.Tensor(shape=(self.rank, self.video_hidden + 1, self.output_dim), dtype=ms.float32, init=XavierNormal()))
        self.text_factor = ms.Parameter(ms.Tensor(shape=(self.rank, self.text_out + 1, self.output_dim), dtype=ms.float32, init=XavierNormal()))
        self.fusion_weights = ms.Parameter(ms.Tensor(shape=(1, self.rank), dtype=ms.float32, init=XavierNormal()))
        self.fusion_bias = ms.Parameter(ms.Tensor(shape=(1, self.output_dim), dtype=ms.float32, init=Zero()))

        #init teh factors
        # self.audio_factor = initializer(XavierNormal(), self.audio_factor)
        # self.video_factor = initializer(XavierNormal(), self.video_factor)
        # self.text_factor = initializer(XavierNormal(), self.text_factor)
        # self.fusion_weights = initializer(XavierNormal(), self.fusion_weights)
        # self.fusion_bias.set_data(ops.Fill(self.fusion_bias.data, 0))

    def construct(self, audio_x, video_x, text_x):
        '''
        :param audio_x: tensor of shape (batch_size, audio_in)
        :param video_x: tensor of shape (batch_size, video_in)
        :param text_x:  tensor of shape (batch_size, sequence_len, text_in)
        :return:
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product

        # todo: 添加gpu
        ones = ops.Ones()
        concat = ops.Concat(1)
        _audio_h = ms.Parameter(ones((batch_size, 1), audio_h.dtype), requires_grad=False)
        _audio_h = concat((_audio_h, audio_h))
        _video_h = ms.Parameter(ones((batch_size, 1), video_h.dtype), requires_grad=False)
        _video_h = concat((_video_h, video_h))
        _text_h = ms.Parameter(ones((batch_size, 1), text_h.dtype), requires_grad=False)
        _text_h = concat((_text_h, text_h))

        fusion_audio = ops.matmul(_audio_h, self.audio_factor)
        fusion_video = ops.matmul(_video_h, self.video_factor)
        fusion_text = ops.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # squeeze可能有问题，matmul输出的tensor不能squeeze没有补全
        output = ops.matmul(self.fusion_weights, ops.transpose(fusion_zy, (1, 0, 2))).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = ops.softmax(output)
        return output