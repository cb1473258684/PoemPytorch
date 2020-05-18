from time import strftime,clock
import multiprocessing
import linecache
import tqdm
from poem.poemtorch.Encoder import EncoderPoem
from poem.poemtorch.Decoder import AttnDecoder
import torch as t
from torch.utils import data
import torch.nn.functional as F
import numpy as np
USE_CUDA = t.cuda.is_available()
device = t.device("cpu")
# device = t.device("cuda" if USE_CUDA else "cpu")
from poem.poemtorch.loadData import ShipData
def display(*args):
    '''display data'''
    print(strftime('[%H:%M:%S]'),end=' ')
    print(*args)

def runFuntion(func,*args):
    start = clock()
    res = func(*args)
    end = clock()
    print("{} function spend time: {}s".format(func.__name__,(end-start)))
    return res

##########################################
#将../processfile.txt文件分三个线程分别写到训练集、测试集、验证集中
##########################################
def task(name,Rfile,Wfile,line,share_lock):

    print("{} start".format(name),end='\t')
    while line.value <= 30088:
        text = linecache.getline(Rfile,line.value)
        share_lock.acquire()
        line.value += 1
        share_lock.release()
        if len(text) == 0:
            continue
        a = text.split(sep='\t')
        a.insert(len(a) // 2, ',')
        text = " ".join(a)
        with open(Wfile,'a',encoding='utf-8') as f:
            f.writelines(text)
    print("{} end".format(name))
# share_lock = multiprocessing.Manager().Lock()
# line_ptr = multiprocessing.Manager().Value('i',0)

# process = []
# process_one = multiprocessing.Process(target=task,args=('process_one','../processfile.txt','./train.txt',line_ptr,share_lock))
# process_two = multiprocessing.Process(target=task, args=('process_two','../processfile.txt', './test.txt', line_ptr, share_lock))
# process_there = multiprocessing.Process(target=task, args=('process_there','../processfile.txt', './valid.txt', line_ptr, share_lock))
# process.append(process_one)
# process.append(process_two)
# process.append(process_there)
# for p in process:
#     p.start()
# for p in process:
#     p.join()
##################################################
#接下来构建词袋wordToindex,indexToword,
##################################################
def BuildVoc(file,wordToindex,indexToword,num_words):
    with open(file,mode='r',encoding='utf-8') as f:
        for line in f:
            seq_in,seq_out = line.strip('\n').split(sep=',')
            seq = seq_in.split(' ')
            seq.remove('')
            seq_out = seq_out.split(' ')
            seq_out.remove('')
            words  = seq+seq_out
            for word in words:
                if word not in wordToindex:
                    wordToindex[word] = num_words
                    indexToword[num_words] = word
                    num_words += 1
    return num_words



def test():
    a = t.tensor([[5, 6, 3], [7, 5, 1], [1, 2, 8]], dtype=float)

    print("a = {}".format(a))
    l = F.log_softmax(a, dim=1)
    print("log_softmax(a) = {}".format(l))

    s = F.softmax(a, dim=1)
    print("softmax(a) = {}".format(s))
    b = t.tensor([0, 1, 2])

    i = l.gather(dim=1, index=t.tensor([[0], [1], [2]]))
    print(sum(i) / 3)

    print(s[1, :].sum())

    c = F.cross_entropy(a, b)

    print("c = {}".format(c))

    d = t.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    loss = t.nn.NLLLoss()
    print(loss(l, b))
if __name__ == '__main__':
    # 默认词向量
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    wordToindex = {"PAD":PAD_token,  "SOS":SOS_token,  "EOS":EOS_token}
    indexToword = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
    num_words = 3
    files = ['./test.txt','./train.txt','valid.txt']


    # for file in tqdm.tqdm(files):
    #     num_words = BuildVoc(file,wordToindex,indexToword,num_words)
    # np.save('num_words',num_words)
    # np.save('wordToindex',wordToindex)
    # np.save('indexToword', indexToword)
    def my_collate(batch):
        batch.sort(key=lambda data: len(data[0]), reverse=True)
        lengths_seq_in = []
        lengths_seq_out = []
        seq_x = []
        seq_y = []
        for (seq_in,seq_out) in batch:
            lengths_seq_in.append(len(seq_in))
            lengths_seq_out.append(len(seq_out))
            seq_x.append(seq_in)
            seq_y.append(seq_out)
        max_len_x = max(lengths_seq_in)
        max_len_y = max(lengths_seq_out)
        seq_x = np.array([np.pad(s, (0, max_len_x-len(s)), 'constant') for s in seq_x])
        seq_y = np.array([np.pad(s, (0, max_len_y-len(s)), 'constant') for s in seq_y])
        mask = np.zeros_like(seq_y)
        indexNO1 = seq_y > 0
        mask[indexNO1] = 1
        return t.LongTensor(seq_x).to(device=device),\
               t.LongTensor(seq_y).to(device=device),\
               t.tensor(lengths_seq_in,dtype=float).to(device=device),\
               t.tensor(lengths_seq_out).to(device=device),\
               t.ByteTensor(mask).to(device=device)

    test_data = ShipData("./test.txt")
    test_iter = data.DataLoader(test_data,batch_size=8,drop_last=True,collate_fn=my_collate)

    num_words = np.load('num_words.npy').item()
    embedding = t.nn.Embedding(num_words,embedding_dim=10).to(device=device)

    enc = EncoderPoem(hiddenSize=10,embedding=embedding,n_layers=2)
    dec = AttnDecoder(atten_model='dot',embedding=embedding,hiddenSize=10,output_size=num_words,n_layers=2)
    # a = embedding(t.tensor([[1,2,3],[2,1,3]]))

    for seq_in,seq_out,lengths_seq_in,lengths_seq_out,mask in test_iter:
        print("x = {}".format(seq_in))
        print("y = {}".format(seq_out))
        print("lengths_x = {}".format(lengths_seq_in))
        print("lengths_y = {}".format(lengths_seq_out))
        print("mask = {}".format(mask))
        outputs,hidden = enc(seq_in,lengths_seq_in)
        s = t.ones(size=(8,1)).long()
        output_dec, hidden_dec = dec(s,hidden[:2,:,:],outputs)
        a = seq_out[:,0].view(-1,1)
        l = -t.log(t.gather(output_dec,dim=1,index=a)).squeeze(1)
        mask = mask[:,0]
        b = l.masked_select(mask=mask).mean()

        break











