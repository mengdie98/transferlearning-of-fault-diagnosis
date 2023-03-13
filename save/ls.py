import os
import pandas as pd
import scipy.io
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import pywt
import multiprocessing as mp

def GetT_FDomainGraph(name, df, path, p):

    df = [df[a] for a in df.columns if 'FE' in a]
    for a in range(0,len(df[0]),128):
        df_t = [(df[0][a:a+512], np.arange(1, 512), 'morl')]
        result = p.starmap(pywt.cwt, df_t)
        fig = plt.figure(figsize=[6, 24], dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.imshow(result[0][0])
        plt.set_cmap('hot')
        plt.axis('off')
        plt.colorbar(shrink=0.5).remove()
        
        save_path = path.replace('CWRUData', 'CWRUData-picture')
        filename = os.path.splitext(name)[0]+f'_{a/128}'+'.png'
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close(fig)

if __name__ == '__main__':
    path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1730\7'
    # save_path_t = r'D:\python_workfile\ML_Classify\graph'
    # save_path_f = r'D:\python_workfile\ML_Classify\graph'
    save_path_tf = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData-picture\12K_Drive_End\1730\7'
    # fault = 'baseline'
    # speed = ''
    # load = ''
    # word_list = [fault, speed, load]
    # time_stat_namelist = ['']
    # time_stat_chanlist = [1, 2, 3, 4, 5, 6, 7]
    time_stat = pd.DataFrame()

    p = mp.Pool(processes=12)

    for filepath, dirnames, filenames in os.walk(path):
        # 获取每个数据文件的时域特征值、频谱图、时频图
        for filename in filenames:

            
            if os.path.splitext(filename)[1] == '.csv':
                SourceData = pd.read_csv(os.path.join(filepath, filename))
                SourceData = SourceData.loc[:, ~SourceData.columns.str.contains('^Unnamed')]
                
                GetT_FDomainGraph(filename, SourceData, filepath, p)
            elif os.path.splitext(filename)[1] == '.mat':
                data = scipy.io.loadmat(os.path.join(filepath, filename))
                SourceData = pd.DataFrame()
                for key in data:
                    if not key.startswith('__'):
                        SourceData[key] = pd.Series(data[key].flatten())
                GetT_FDomainGraph(filename, SourceData, filepath, p)
                
                
# import scipy.io
# import pandas as pd

# # 加载 .mat 文件
# data = scipy.io.loadmat('D:\save data\Python\毕设\\12k_Drive_End_OR007@3_3_147.mat')

# # 创建一个空数据框
# df = pd.DataFrame()

# # 循环遍历每个变量并添加到数据框
# for key in data:
#     if not key.startswith('__'):
#         df[key] = pd.Series(data[key].flatten())

# # 将数据框保存为 CSV 文件
# df.to_csv('D:\save data\Python\毕设\\12k_Drive_End_OR007@3_3_147.csv', index=False)
