import numpy as np
from scipy import interpolate
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tools.文本读取 import read_simple
from data.compute.respeak_simple.resuint import resunit

print(torch.__version__)
print(torch.cuda.is_available())
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签，如果想要用新罗马字体，改成 Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def write(filename, files, values):
    file = open(filename, 'w')
    for k, v in zip(files, values):
        file.write(str(k) + " " + str(v) + "\n")
    file.close()

import torch.nn as nn
import torch

import torch.nn.functional as F

def draw_set(title,xlabel="Raman shift ${(cm^{-1}})}$",ylabel="Intensity (a.u.)"):

    font_title=dict(fontsize=16,
              color='k',
              family='Times New Roman',
              weight='bold',
              style='normal',
              )        #设置标题字体、字号、颜色等

    font_dict=dict(fontsize=14,
              color='k',
              family='Times New Roman',
              weight='bold',
              style='normal',
              )        #设置其他字体、字号、颜色等

    plt.title(title,fontdict=font_title)
    plt.xlabel(xlabel,fontdict=font_dict)
    plt.ylabel(ylabel,fontdict=font_dict)

    #刻度线
    plt.tick_params(axis='x', labelsize=8, pad=5, direction='out') #刻度线标签大小距离，刻度线方向等
    plt.tick_params(axis='y', labelsize=8, pad=5, direction='out')#刻度线标签大小距离，刻度线方向等
    plt.xticks(fontproperties='Times New Roman', size=12, weight='bold') #设置刻度线标签字体大小及加粗
    plt.yticks(fontproperties='Times New Roman', size=12, weight='bold')

    #设置公式中的字体
    plt.rcParams['mathtext.fontset'] = 'stix' #上标字体
    plt.rcParams['mathtext.default']='bf'     #上标加粗

    plt.tight_layout()#防止字被遮挡


#定义存储特征和梯度的数组
fmap_block = list()
grad_block = list()

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())
    print("backward_hook:",grad_in[0].shape,grad_out[0].shape)

def farward_hook(module, input, output):
    fmap_block.append(output)
    print("farward_hook:",input[0].shape,output.shape)


def Grad_CAM(model_path ='2_1.bin',data_path = 'data-需要全谱归一化并且截取250~4000波数作为考虑区域'):
    map1=[]
    mapall=[0]*2800
    for p in range(8):
        #加载模型
        model = torch.load(model_path,map_location='cpu')
        model.eval()  # 评估模式
    
        # 注册hook
        fh=model.conv_3.register_forward_hook(farward_hook) #model后是网络层的名字
        bh=model.conv_3.register_backward_hook(backward_hook) #
    
        #加载变量并进行预测
        n,m1,name = read_simple(data_path,0,2800) #数据路径，数据的起始行和中止行
        n1= np.expand_dims(n, axis=1)
        n1= np.expand_dims(n1, axis=1)
        n1=torch.tensor(n1).cpu().float()
        preds=model(n1[1])
        # preds=[]
        # for t in range(len(n)):
        
        #     preds1=model(n1[t])
        #     preds1=preds1.detach().cpu().numpy()
        #     preds.append(preds1)
        #     print(name[t])
        
        
        
        lb=[0.,0.,0.,0.,0.,0.,0.,0.]
        lb[p]=1.
        print(lb)
        
    
        #构造label，并进行反向传播
        trues = torch.tensor([lb]).cpu().float()
        ce_loss=nn.BCEWithLogitsLoss()
        loss=ce_loss(preds,trues)
        loss.backward()
    
        # 卸载hook
        fh.remove()
        bh.remove()
    
        #获取特征图和梯度
        layer1_grad=grad_block[-1][0]
        layer1_fmap=fmap_block[-1][0]
    
        #计算权重
        cam=layer1_grad[0].mul(layer1_fmap[0])
        for i in range(1,layer1_grad.shape[0]):
            cam+=layer1_grad[i].mul(layer1_fmap[i])
    
        cam = cam.detach().cpu().numpy()
    
        #插值
        x = np.linspace(1,2800,cam.shape[0])
        tck = interpolate.splrep(x,cam)
        xx = np.linspace(min(x),max(x),2800)
        map = interpolate.splev(xx,tck,der=0)
        map1.append(map)
        mapall=mapall+map
        

    print(mapall)
    mapp=map1[6]-((mapall-map1[6])/7)

    plt.plot(m1[0],mapp,c='g')
    plt.xlim(m1[0].min(),m1[0].max())
    draw_set('')
    plt.show()

    plt.plot(m1[0],n[1],c='b')
    plt.xlim(m1[0].min(),m1[0].max())
    draw_set('')
    plt.show()
    write( '582_6.txt', m1[0], mapp)
    return mapp


if __name__ == '__main__':
    model_path = '2_1.bin'
    data_path = 'data-需要全谱归一化并且截取250~4000波数作为考虑区域'
    mapp = Grad_CAM(model_path,data_path)
    # preds=np.array(preds)  
    # preds=preds.reshape(108,8)
