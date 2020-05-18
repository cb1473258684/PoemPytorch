import numpy as np
import os
import random
import tqdm
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.switch_backend('agg')
import matplotlib.ticker as ticker
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
class Seqseq:
    def __init__(self,encoder,decoder,vis,save_dir):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.vis = vis
        self.wordToindex = np.load('wordToindex.npy').item()
        self.indexToword = np.load('indexToword.npy').item()
        self.num_words = np.load('num_words.npy').item()

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),lr=0.001)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),lr=0.001)
        self.save_dir = save_dir

    def train(self,seq_in,seq_out,lengths_seq_in,mask,lengths_seq_out,clip,teacher_forcing_ratio):
        # 零化梯度
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        #一个batch的训练

        loss = 0
        n_totals = 0
        print_loss = []
        batch_size,max_length= seq_out.size()

        outputs, hidden = self.encoder(seq_in, lengths_seq_in)

        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]]).view(batch_size,1).to(device)
        decoder_hidden = hidden[:self.decoder.n_layers, :, :]

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for t in range(max_length):
                decoder_output, decoder_hidden,atten_weights = self.decoder(decoder_input, decoder_hidden,outputs)
                #使用目标seq_out作为输入
                decoder_input = seq_out[:,t].view(batch_size, 1).detach()

                # 计算loss
                t_loss, n_total = self.Calcloss(decoder_output, seq_out[:, t], mask[:, t])
                loss += t_loss
                print_loss.append(t_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(max_length):
                decoder_output,decoder_hidden,atten_weights = self.decoder(decoder_input,decoder_hidden)
                _,topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).view(batch_size,1).to(device).detach()

                #计算loss
                t_loss,n_total = self.Calcloss(decoder_output,seq_out[:,t],mask[:,t])
                loss += t_loss
                print_loss.append(t_loss.item() * n_total)
                n_totals += n_total

        loss.backward()

        # 剪辑梯度：梯度被修改到位
        _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # 调整模型权重
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_loss) / n_totals

    def trainIters(self,data_iter,clip,teacher_forcing_ratio,epochs):
        for epoch in tqdm.trange(1,epochs):
            for seq_in,seq_out,lengths_seq_in,lengths_seq_out,mask in data_iter:
                loss = self.train(seq_in,seq_out,lengths_seq_in,mask,lengths_seq_out,clip,teacher_forcing_ratio)
                self.vis.plot('loss',loss)
            tqdm.tqdm.write('epoch : {} loss : {}'.format(epoch, loss))
            self.vis.log(info="epoch : {} ,loss : {}".format(epoch, loss))
            if epoch % 20 == 0:
                self.SaveModel(self.save_dir,'poem',epoch,loss)

    def Calcloss(self,pre,target,mask):
        crossentropy = -torch.log(torch.gather(pre,dim=1,index=target.view(-1,1))).squeeze(1)
        loss = crossentropy.masked_select(mask=mask).mean()
        return loss,mask.sum().item()

    def showAttention(self,input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([' ']+['<SOS>']+input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # plt.show()
        self.vis.vis.matplot(plot=plt)

    def GetIndex(self,input):
        input = input.strip('\n').split(' ')
        input = list(filter(None, input))
        input.insert(0, "SOS")
        input.append("EOS")
        poem_in = [self.wordToindex[word] for word in input]
        return poem_in,len(poem_in)

    def evaluate(self,input_sentence,max_length = 7):
        '''
        :param input_sentence: str = "白日 依山 尽"
        :return: 下一句诗词
        '''

        with torch.no_grad():
            seq_in,lengths_seq_in = self.GetIndex(input_sentence)
            max_length = len(seq_in)

            seq_in = torch.LongTensor(seq_in).view(1,-1).to(device)
            lengths_seq_in = torch.tensor([lengths_seq_in],dtype=float)

            output = []
            decoder_attentions = torch.zeros(max_length, max_length)

            outputs, hidden = self.encoder(seq_in, lengths_seq_in)

            decoder_input = torch.LongTensor([[SOS_token]]).view(1, 1).to(device=device)
            decoder_hidden = hidden[:self.decoder.n_layers, :, :]

            for t in range(max_length):
                decoder_output,decoder_hidden,atten_weights = self.decoder(decoder_input,decoder_hidden,outputs)
                _,topi = decoder_output.topk(1)
                topi = topi.item()
                decoder_input = torch.LongTensor([[topi]]).view(1, 1).to(device=device)
                #保存输出和注意力权重
                output.append(self.indexToword[topi])
                decoder_attentions[t] = atten_weights.data
                if topi == EOS_token:
                    break;
            return output,decoder_attentions


    def evaluateAndShowAttention(self, input_sentence):
        output_words, attentions = self.evaluate(input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        self.showAttention(input_sentence, output_words, attentions)

    def SaveModel(self,dir,model_name,epoch,loss):
        directory = os.path.join(dir, model_name,
                                 '{}-{}_{}'.format(self.encoder.n_layers, self.decoder.n_layers, self.decoder.hiddenSize))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'en': self.encoder.state_dict(),
            'de': self.decoder.state_dict(),
            'en_opt': self.encoder_optimizer.state_dict(),
            'de_opt': self.decoder_optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))

    def LoadModel(self,loadFilename):
        checkpoint = torch.load(loadFilename)
        self.encoder.load_state_dict(checkpoint['en'])
        self.decoder.load_state_dict(checkpoint['de'])
        self.encoder_optimizer.load_state_dict(state_dict=checkpoint['en_opt'])
        self.decoder_optimizer.load_state_dict(state_dict=checkpoint['de_opt'])
