import torch
from torch.utils import data
import numpy as np

class ShipData(data.Dataset):
    def __init__(self,file):
        super(ShipData, self).__init__()
        self.data = []
        self.seq_in = []
        self.seq_out = []

        self.wordToindex = np.load('wordToindex.npy').item()
        self.indexToword = np.load('indexToword.npy').item()
        self.num_words = np.load('num_words.npy').item()

        with open(file,mode='r',encoding='utf-8') as f:
            for line in f:
                s_in, s_out = line.strip('\n').split(sep=',')
                s_in = s_in.split(' ')
                s_in.remove('')
                s_out = s_out.split(' ')
                s_out.remove('')

                s_in.insert(0,"SOS")
                s_in.append("EOS")
                s_out.insert(0, "SOS")
                s_out.append("EOS")

                self.data.append((" ".join(s_in)," ".join(s_out)))
                self.seq_in.append(s_in)
                self.seq_out.append(s_out)
    def __getitem__(self, item):
        poem_in = [self.wordToindex[word] for word in self.seq_in[item]]
        poem_out = [self.wordToindex[word] for word in self.seq_out[item]]

        return poem_in,poem_out
    def __len__(self):
        return len(self.data)


