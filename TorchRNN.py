from torch.utils import data
import json
import torch.nn as nn
import torch
import numpy as np
class ShipDataset(data.Dataset):
    def __init__(self,vali = 'dataSetOne.npz',Train = True):
        super(ShipDataset, self).__init__()
        #获取数据清单
        with open('datalist.json', 'r', encoding='utf-8') as fjson:
            datadict = json.load(fjson)

        # data = ['dataSetOne.npz','dataSetTwo.npz','dataSetThere.npz','dataSetFour.npz'
        #              ,'dataSetFive.npz','dataSetSix.npz','dataSetSeven.npz','dataSetEight.npz'
        #              ,'dataSetNine.npz','dataSetTen.npz']

        self.valid = {vali:datadict[vali]}
        del datadict[vali]
        self.train = datadict

        self.ModeTrain = Train

    def __getitem__(self, item):
        train = sorted(self.train.items(), key=lambda x: x[1])
        # dataSetOne.npz  128184        dataSetFour.npz     128715
        # dataSetTwo.npz     128534      dataSetThere.npz  128039
        # dataSetFive.npz    127958      dataSetSix.npz      127759
        # dataSetSeven.npz   127277      dataSetEight.npz    127917
        # dataSetNine.npz    127969      dataSetTen.npz      127421

    #获取数据并返回
        if self.ModeTrain:
            if item < train[0][1]:
                npz_data = np.load(train[0][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']

            elif train[1][1] + train[0][1] > item >= train[0][1]:
                npz_data = np.load(train[1][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']
                item = item-train[0][1]
            elif train[2][1] + train[1][1] + train[0][1] > item >= train[1][1] + train[0][1]:
                npz_data = np.load(train[2][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']
                item = item-train[1][1] - train[0][1]
            elif train[3][1] + train[2][1] + train[1][1] + train[0][1] > item >= train[2][1] + train[1][1] + train[0][
                1]:
                npz_data = np.load(train[3][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']
                item = item-train[2][1] - train[1][1] - train[0][1]
            elif train[4][1] + train[3][1] + train[2][1] + train[1][1] + train[0][1] > item >= train[3][1] + train[2][
                1] + train[1][1] + train[0][1]:
                npz_data = np.load(train[4][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']
                item = item - train[3][1] - train[2][1] - train[1][1] - train[0][1]

            elif train[5][1] + train[4][1] + train[3][1] + train[2][1] + train[1][1] + train[0][1] > item >= train[4][
                1] + train[3][1] + train[2][1] + train[1][1] + train[0][1]:
                npz_data = np.load(train[5][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']
                item = item - train[4][1] - train[3][1] - train[2][1] - train[1][1] - train[0][1]

            elif train[6][1] + train[5][1] + train[4][1] + train[3][1] + train[2][1] + train[1][1] + train[0][
                1] > item >= train[5][1] + train[4][1] + train[3][1] + train[2][1] + train[1][1] + train[0][1]:
                npz_data = np.load(train[6][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']
                item = item-train[5][1] - train[4][1] - train[3][1] - train[2][1] - train[1][1] - train[0][1]

            elif train[7][1] + train[6][1] + train[5][1] + train[4][1] + train[3][1] + train[2][1] + train[1][1] + \
                    train[0][1] > item >= train[6][1] + train[5][1] + train[4][1] + train[3][1] + train[2][1] + \
                    train[1][1] + train[0][1]:
                npz_data = np.load(train[7][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']
                item = item -train[6][1] -train[5][1] - train[4][1] - train[3][1] - train[2][1] - train[1][1] - train[0][1]

            elif train[8][1] + train[7][1] + train[6][1] + train[5][1] + train[4][1] + train[3][1] + train[2][1] + \
                    train[1][1] + train[0][1] > item >= train[7][1] + train[6][1] + train[5][1] + train[4][1] + \
                    train[3][1] + train[2][1] + train[1][1] + train[0][1]:
                npz_data = np.load(train[8][0])
                data_X = npz_data['data_X']
                data_Y = npz_data['data_Y']
                item = item - train[7][1]-train[6][1] -train[5][1] - train[4][1] - train[3][1] - train[2][1] - train[1][1] - train[0][1]

            list_data = [np.load(str) for str in self.train]
            data_X = [list_data[i]['data_X'] for i in range(len(list_data))]
            data_Y = [list_data[i]['data_Y'] for i in range(len(list_data))]
        else:
            valid = list(self.valid.keys())
            npz_data=np.load(valid[0])
            data_X = npz_data['data_X']
            data_Y = npz_data['data_Y']


        return data_X[item], data_Y[item]
    def __len__(self):
    #返回数据的数量
        if self.ModeTrain:
            num = self.train.values()
            return sum(num)-800000
        else:
            valid = list(self.valid.values())
            return valid[0]


class RNNModel(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers):
        super(RNNModel, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size,hidden_size,n_layers,batch_first=True)
        self.linear = nn.Linear(self.hidden_size,input_size)
    def forward(self, input,state=None):
        batch,_,_ = input.size()
        if state is None:
            h = torch.randn(self.n_layers,batch,self.hidden_size).float()
            c = torch.randn(self.n_layers,batch,self.hidden_size).float()
        else:
            h,c=state
        # output [batchsize,time,hidden_size]
        output,state = self.rnn(input,(h,c))

        # output = self.linear(output)

        return output,state

def Main():
    hidden_size =100
    inputsize = 100
    layers = 2
    epochs = 50
    train_data = ShipDataset()
    train_loader = data.DataLoader(train_data, batch_size=4, num_workers=0, shuffle=False)
    model = RNNModel(inputsize,hidden_size,layers)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for i,trainset in enumerate(train_loader):
            X,Y=trainset
            X,Y = X.float(),Y.float()
            # print(X.size())
            # print(Y.size())
            # print(Y[:,0,:].long())
            out_y,_ = model(X)
            # print(out_y)
            # y = out_y[:,-1,:]
            loss = criterion(out_y, Y)
            # loss = torch.zeros(1,1).float()
            # for i in range(X.size(0)):
            #     loss = loss+criterion(out_y[i],Y[i])
            # loss = loss/X.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss)
        torch.save({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            'loss': loss,
        }, 'Poemcheckpoint.tar')
    # dataiter = iter(train_loader)
    # X,Y = next(dataiter)
    # print(X)
    # batch,time,inputsize = X.size()
    # print(batch)
    # out,state = model(X.float())
    # print(out)


if __name__ == '__main__':
    Main()