import zipfile
import json
import os
import re
from glob import glob
from gensim.models import word2vec
import numpy as np
'''
实现方式：RNN，诗词上半句为输入x，下半句为输出为label y
'''

#unzip poem
def unzip_poem(zipfilePath):
    if os.path.exists('poem_data'):
        pass
    else:
        os.mkdir("poem_data")
    file_zip = zipfile.ZipFile(zipfilePath,'r')
    for file in file_zip.namelist():
        file_zip.extract(file,r'./poem_data')
    file_zip.close()

def SavePoem(file,Poemfiles):
    for filepath in Poemfiles:
        with open(filepath,'r',encoding='utf-8') as f:
            pdict = json.load(f)
            Onefile = ["".join(l["paragraphs"]) for l in pdict]

        with open(file,'a+',encoding="utf-8") as f:
            for data in Onefile:
                f.write(data)
                f.write('\n')

#remove：□、（）   if only one sentence in a poem，remove the poem.
#divide word using 。
def processPoem(file,processfile):
    data = []
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            line = re.sub("[\□\{\}\罒\/\%\[\]\（\）]", "", line.strip('\n'))
            if(line.count('，') == line.count('。') and len(line) != 0 ):  #需要的是两句以上的句子
                line = line.replace('，',"")
                listend = [i.start() for i in re.finditer('。', line)]
                listend.insert(0, 0)
                for i in range(len(listend) - 1):
                    start = listend[i]
                    end = listend[i + 1]
                    data.append(line[start:end].replace('。', ""))


    with open(processfile,'w+',encoding='utf-8') as fw: #将一个处理好的诗句写入文件
        for line in data:
            l=[]
            start = 0
            if len(line) == 14:
                l = [2,2,3,2,2,3]
            elif len(line) == 10:
                l = [2,2,1,2,2,1]
            elif len(line) == 12:
                l = [2,2,2,2,2,2]
            elif len(line) == 8:
                l = [2,2,2,2]
            elif len(line) == 6:
                l = [1,1,1,1,1,1]
            else:
                pass
            for i in range(len(l)):
                fw.write(line[start:start + l[i]])
                if i != len(l)-1:
                    fw.write('\t')
                elif line == data[-1]:
                    pass
                else:
                    fw.write('\n')
                start = start + l[i]


def GetFILE():
    Poemfiles = []

    for root,dirs,files in os.walk(r'.'):
        file_pattern = os.path.join(root,"poet*.json")
        for f in glob(file_pattern):
            Poemfiles.append(f)

    return Poemfiles

def Word2Vec_poem(file,target):
    sentense = word2vec.Text8Corpus(file)
    model = word2vec.Word2Vec(sentense, sg=1, size=100, window=5, min_count=0, negative=3, sample=0.001, hs=1,
                              workers=4)
    model.save(target)
    return
#十字交叉验证，即将数据随机分成十份，每次取其中九份作为训练集，剩下的作为验证集
def cross_validation(file):
    feature_one = np.matrix((1,1,1))
    label_one = []

    feature_two = []
    label_two = []

    feature_there = []
    label_there = []

    feature_four = []
    label_four = []

    feature_five = []
    label_five = []

    feature_six = []
    label_six = []

    feature_seven = []
    label_seven = []

    feature_eight = []
    label_eight = []

    feature_nine = []
    label_nine = []

    feature_ten = []
    label_ten = []
    VecPoem = word2vec.Word2Vec.load('PoemVec')
    with open(file, 'r', encoding='utf-8') as fp:
        for linestr in fp:
            linestr = linestr.strip('\n')
            list_line = linestr.split("\t")
            data_feature = np.zeros((3,100),dtype=float)  #这里设置的100维是根据前面word2vec将每个词变为100维向量
            data_label = np.zeros((3,100),dtype=float)


            data_feature[0:len(list_line) // 2] = VecPoem[[str for str in list_line[0:len(list_line) // 2]]]
            data_label[0:len(list_line) //2] = VecPoem[[str for str in list_line[len(list_line)//2:len(list_line)]]]

            data_feature = np.matrix(data_feature,dtype=float)
            data_label = np.matrix(data_label,dtype=float)

            chance = np.random.randint(100)
            if chance >= 90:
                feature_one.append(data_feature)
                label_one.append(data_label)
            elif 90>chance>=80:
                feature_two.append(data_feature)
                label_two.append(data_label)
            elif 80>chance>=70:
                feature_there.append(data_feature)
                label_there.append(data_label)
            elif 70>chance>=60:
                feature_four.append(data_feature)
                label_four.append(data_label)
            elif 60>chance>=50:
                feature_five.append(data_feature)
                label_five.append(data_label)
            elif 50>chance>=40:
                feature_six.append(data_feature)
                label_six.append(data_label)
            elif 40>chance>=30:
                feature_seven.append(data_feature)
                label_seven.append(data_label)
            elif 30>chance>=20:
                feature_eight.append(data_feature)
                label_eight.append(data_label)
            elif 20>chance>=10:
                feature_nine.append(data_feature)
                label_nine.append(data_label)
            else:
                feature_ten.append(data_feature)
                label_ten.append(data_label)

    state = np.random.get_state()
    np.random.shuffle(feature_one)
    np.random.shuffle(feature_two)
    np.random.shuffle(feature_there)
    np.random.shuffle(feature_four)
    np.random.shuffle(feature_five)
    np.random.shuffle(feature_six)
    np.random.shuffle(feature_seven)
    np.random.shuffle(feature_eight)
    np.random.shuffle(feature_nine)
    np.random.shuffle(feature_ten)
    np.random.set_state(state)
    np.random.shuffle(label_one)
    np.random.shuffle(label_two)
    np.random.shuffle(label_there)
    np.random.shuffle(label_four)
    np.random.shuffle(label_five)
    np.random.shuffle(label_six)
    np.random.shuffle(label_seven)
    np.random.shuffle(label_eight)
    np.random.shuffle(label_nine)
    np.random.shuffle(label_ten)

    #写一个数据清单
    with open('datalist.json','w',encoding='utf-8') as fjson:
        d_json = {'dataSetOne':len(feature_one),'dataSetTwo':len(feature_two),'dataSetThere':len(feature_there)
                  ,'dataSetFour':len(feature_four),'dataSetFive':len(feature_five),'dataSetSix':len(feature_six)
                  ,'dataSetSeven':len(feature_seven),'dataSetEight':len(feature_eight),'dataSetNine':len(feature_nine)
                  ,'dataSetTen':len(feature_ten)}
        json.dump(d_json,fjson)

    np.savez('dataSetOne',data_X = feature_one,data_Y = label_one)
    np.savez('dataSetTwo', data_X=feature_two, data_Y=label_two)
    np.savez('dataSetThere', data_X=feature_there, data_Y=label_there)
    np.savez('dataSetFour', data_X=feature_four, data_Y=label_four)
    np.savez('dataSetFive',data_X=feature_five, data_Y=label_five)
    np.savez('dataSetSix', data_X=feature_six, data_Y=label_six)
    np.savez('dataSetSeven', data_X=feature_seven, data_Y=label_seven)
    np.savez('dataSetEight', data_X=feature_eight, data_Y=label_eight)
    np.savez('dataSetNine', data_X=feature_nine, data_Y=label_nine)
    np.savez('dataSetTen', data_X=feature_ten, data_Y=label_ten)


def Main():
    # vali ='dataSetOne.npz'
    # # data = [np.load(str) for str in vali]
    # # data_X = data[0]['data_X']
    # data = ['dataSetOne.npz', 'dataSetTwo.npz']
    # data.remove(vali)
    #
    # list_data = [np.load(str) for str in data]
    # list_data = [np.load(str) for str in data]
    # data_X = [list_data[i]['data_X'] for i in range(len(list_data))]
    # data_Y = [list_data[i]['data_Y'] for i in range(len(list_data))]
    # d = data_X[0]+data_X[1]
    # print(len(d))
    # dataSetOne.npz  128184        dataSetFour.npz     128715
    #dataSetTwo.npz     128534      dataSetThere.npz  128039
    #dataSetFive.npz    127958      dataSetSix.npz      127759
    #dataSetSeven.npz   127277      dataSetEight.npz    127917
    #dataSetNine.npz    127969      dataSetTen.npz      127421
    with open('datalist.json', 'w', encoding='utf-8') as fjson:
        d_json = {'dataSetOne.npz': 128184, 'dataSetTwo.npz': 128534, 'dataSetThere.npz': 128039
            , 'dataSetFour.npz': 128715, 'dataSetFive.npz': 127958, 'dataSetSix.npz': 127759
            , 'dataSetSeven.npz': 127277, 'dataSetEight.npz': 127917, 'dataSetNine.npz': 127969
            , 'dataSetTen.npz': 127421}
        json.dump(d_json, fjson)
    # with open('datalist.json', 'r', encoding='utf-8') as fjson:
    #     datalist = json.load(fjson)

    # d = {'a':1,'sss':2}
    # a = d.values()
    # print(a[0])
    # cross_validation(r'processfile.txt')
    # count =0;
    # data = []
    # with open('processfile.txt','r',encoding='utf-8') as fp:
    #     for linestr in fp:
    #         linestr = linestr.strip('\n')
    #         list_line = linestr.split("\t")
    #         data_feature = np.zeros((3, 100),dtype=float)
    #         data_label = np.zeros((3, 100), dtype=float)
    #         print(list_line[0:len(list_line)//2])
    #
    #         print(list_line[len(list_line)//2:len(list_line)])
    #     # Word2Vec_poem(r'processfile.txt',r'./PoemVec')
    #         VecPoem = word2vec.Word2Vec.load('PoemVec')
    #         data_feature[0:len(list_line)//2] = VecPoem[[str for str in list_line[0:len(list_line)//2]]]
    #         data_label[(len(list_line) // 2):len(list_line)] = VecPoem[[str for str in list_line[(len(list_line) // 2):len(list_line)]]]
    #         data.append(data_feature)
    #         count+=1
    #         if count == 2:
    #             break
    #         print(data_label)
    # list_line = ['欲出', '未出', '光辣达','千山', '万山', '如火发']
    # data_label = np.zeros((3, 100), dtype=float)
    # VecPoem = word2vec.Word2Vec.load('PoemVec')
    # data_label[(len(list_line) // 2):-1] = VecPoem[[str for str in list_line[(len(list_line) // 2):len(list_line)]]]
    # print(data_label)
    # print(VecPoem['欲出'])
    # np.save('feature.npy',data)
    # data = np.load('feature.npy')
    # print(data[2])
    # listA = [1,2,3,np.matrix((2,2))]
    # listB = [4,5,6, np.matrix((3,3))]
    # state = np.random.get_state()
    # np.random.shuffle(listA)
    # np.random.set_state(state)
    # np.random.shuffle(listB)
    # np.savez('list.npz',A = listA,B = listB)
    # data = np.load(r'list.npz')
    # print(data['B'][0])



    # a = tf.add(1,1)
    # sess = tf.Session()
    # sess.run(a)
    # sess.close()
    # Poemfiles = GetFILE()
    # SavePoem("test.txt",Poemfiles)
    # processPoem("test.txt",'processfile.txt')
    # strone = "欲出未出光辣达，千山万山如火发。须臾走向天上来，逐却残星赶却月。"
    # strtwo = "欲出未出光辣达千山万山如火发"
    # l = [2,2,3,2,2,3]
    # start = 0
    # for i in range(len(l)):
    #     print(strtwo[start:start+l[i]])
    #     start = start+l[i]
    # with open("t.txt",'w+',encoding="utf-8") as f:
    #     for i in range(5):
    #         f.write("这是测试")
    #         if i != 4:
    #             f.write("\t")
    #         else:
    #             f.write('\n')
    # len_7 =0
    # len_6 = 0
    # len_5 = 0
    # len_4 = 0
    # len_3 = 0
    # len_other = 0
    # with open('processfile.txt', 'r', encoding='utf-8') as f:  # 将一个处理好的诗句写入文件
    #     for line in f:
    #         if len(line.strip('\n')) == 14:
    #             line[0]
    #         elif len(line.strip('\n')) == 10:
    #             len_5+=1
    #         elif len(line.strip('\n')) == 12:
    #             len_6+=1
    #         elif len(line.strip('\n')) == 8:
    #             len_4+=1
    #         elif len(line.strip('\n')) == 6:
    #             len_3+=1
    #         else:
    #             len_other+=1
    #             print(line)
    #
    # print(len_7)
    # print(len_6)
    # print(len_5)
    # print(len_4)
    # print(len_3)
    # print(len_other)

    # data = []
    # x = []
    # y = []
    # strthere = strthere.replace('，',"")
    # print(strthere)
    # liststr = [i.start() for i in re.finditer('。',strthere)]
    # liststr.insert(0,0)
    # print(liststr)
    # for i in range(len(liststr)-1):
    #     start = liststr[i]
    #     end = liststr[i+1]
    #     data.append(strthere[start:end].replace('。',""))
    # print(data)
    # for i in range(2,len(strthere)+1,2):



    # data = []
    # with open(".\poem_data\chinese-poetry-zhCN-master\poetry\poet.song.0.json",'r',encoding='utf-8') as f:
    #     pdict = json.load(f)
    #     print(len(pdict))
    #     for l in pdict:
    #         print(" ".join(l["paragraphs"]))
    #     data.append(pdict[0]["paragraphs"])
    #     print(pdict[0]["paragraphs"])
    #     print(pdict[0]["paragraphs"][0])

if __name__ == '__main__':
    Main()