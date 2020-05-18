'''

'''
from poem.poemtorch.Encoder import EncoderPoem
from poem.poemtorch.Decoder import AttnDecoder
from poem.poemtorch.loadData import ShipData
from poem.poemtorch.visualize import Visualizer
from poem.poemtorch.SeqseqPoem import Seqseq

from torch.utils import data
import numpy as np
import torch as t
USE_CUDA = t.cuda.is_available()
device = t.device("cuda" if USE_CUDA else "cpu")

def my_collate(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)
    lengths_seq_in = []
    lengths_seq_out = []
    seq_x = []
    seq_y = []
    for (seq_in, seq_out) in batch:
        lengths_seq_in.append(len(seq_in))
        lengths_seq_out.append(len(seq_out))
        seq_x.append(seq_in)
        seq_y.append(seq_out)
    max_len_x = max(lengths_seq_in)
    max_len_y = max(lengths_seq_out)
    seq_x = np.array([np.pad(s, (0, max_len_x - len(s)), 'constant') for s in seq_x])
    seq_y = np.array([np.pad(s, (0, max_len_y - len(s)), 'constant') for s in seq_y])
    mask = np.zeros_like(seq_y)
    indexNO1 = seq_y > 0
    mask[indexNO1] = 1
    return t.LongTensor(seq_x).to(device=device), \
           t.LongTensor(seq_y).to(device=device), \
           t.tensor(lengths_seq_in, dtype=float).to(device=device), \
           t.tensor(lengths_seq_out).to(device=device), \
           t.tensor(mask,dtype=t.bool).to(device=device)


if __name__ == '__main__':
    batchSize = 8   #训练的批次大小
    VocSize = 120   #词向量的维度
    epochs = 500   #训练批次轮数
    encoder_n_layers = 2    #编码器层数
    decoder_n_layers = 2    #解码器层数
    hiddenSize = VocSize    #RNN隐藏层向量维度，这里一定要和词向量一致，因为编码器和解码器已经写死了
    atten_mode = ['dot', 'general', 'concat']   #计算解码器的注意力方式，

    clip = 50.0 #梯度裁剪，有利于梯度稳步下降，避免发生梯度爆炸
    teacher_forcing_ratio = 1.0 #使用目标输出引导解码器的概率

    SaveDir = './'  #模型保存路径
    #加载数据
    train_data = ShipData("./train.txt")
    train_iter = data.DataLoader(train_data, batch_size=batchSize, drop_last=True, collate_fn=my_collate)
    #加载词袋大小，构建词向量层
    num_words = np.load('num_words.npy').item()
    embedding = t.nn.Embedding(num_words, embedding_dim=VocSize)
    #初始化编码器、解码器
    enc = EncoderPoem(hiddenSize=hiddenSize, embedding=embedding, n_layers=encoder_n_layers)
    dec = AttnDecoder(atten_model=atten_mode[0], embedding=embedding, hiddenSize=hiddenSize, output_size=num_words, n_layers=decoder_n_layers)

    #初始化可视化工具，Visdom，需要手动输入命令启动：python -m visdom.server -port=1314
    vis = Visualizer(env="poem",port = 1314)

    #初始化模型
    mypoem = Seqseq(encoder=enc,decoder=dec,vis=vis,save_dir=SaveDir)

    #训练
    # mypoem.trainIters(data_iter=train_iter,clip=clip,teacher_forcing_ratio=teacher_forcing_ratio,epochs=epochs)
    mypoem.LoadModel('./poem/2-2_120/140_checkpoint.tar')
    mypoem.evaluateAndShowAttention('愁闻 剑戟 扶危主')