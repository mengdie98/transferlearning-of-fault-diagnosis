import os
import pandas as pd
import scipy.io
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pywt
import multiprocessing as mp
matplotlib.use('pdf')
# len(df[0])-128
def GetT_FDomainGraph(name, df, path, p):
    # for a in range(0,min(len(df[0])-512, 128*512),128):
    #     df_t = [(df[0][a:a+512], np.arange(1, 512), 'morl')]
    #     result = p.starmap(pywt.cwt, df_t)
    #     fig = plt.figure(figsize=[6, 24], dpi=100)
    #     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    #     plt.imshow(result[0][0])
    #     plt.set_cmap('hot')
    #     plt.axis('off')
    #     plt.colorbar(shrink=0.5).remove()
        
    #     # save_path = path.replace('CWRUData', 'CWRUData-picture')
    #     save_path = path.replace('data', 'datapic')
    #     filename = os.path.splitext(name)[0]+f'_{int(a/128)}'+'.png'
    #     if not os.path.exists(os.path.dirname(save_path)):
    #         os.makedirs(os.path.dirname(save_path))
    #     plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', pad_inches=0)
    #     plt.clf()
    #     plt.close(fig)
        
    sample_len = 1024
    sample_rate = 25600
    t = np.arange(0, sample_len/sample_rate, 1/sample_rate)

    wave_name = 'morl'
    total_scal = 256
    fc = pywt.central_frequency(wave_name)
    cparam = 2*fc*total_scal
    scales = cparam/np.arange(total_scal+1, 1, -1)

    for a in range(0, min(len(df[0])-1024,200*512), 200):
        df_t = [(df[0][a:a+sample_len], scales, wave_name)]   #'morl'
        result = p.starmap(pywt.cwt, df_t)    #pywt.cwt, df_t
        fig = plt.figure(figsize=(3, 3))    #以长24英寸 宽6英寸的大小创建一个窗口，一个单位100分辨率
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)   #表示四个方向上的，图表与画布边缘之间的距离

        plt.contourf(t, result[0][1], abs(result[0][0]))
        plt.axis('off')
        plt.colorbar(shrink=0.5).remove()
        save_path = path.replace('CWRUData', 'CWRUDatapic')
        # for i in range(14):
        #     os.makedirs(save_path + '\\' + str(i), exist_ok=True)
        # os.path.splitext用于将文件路径分割成文件名和扩展名两部分，该函数返回一个元组
        filename = os.path.splitext(name)[0]+f'_{int(a/200)}'+'.png'
        if not os.path.exists(os.path.dirname(save_path+'\\')):
            os.makedirs(os.path.dirname(save_path+'\\'))
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close(fig)
        
def main(path):
    p = mp.Pool(processes=36)
    for filepath, dirnames, filenames in os.walk(path):

        for filename in filenames:
            try:
                if os.path.splitext(filename)[1] == '.csv':
                    SourceData = pd.read_csv(os.path.join(filepath, filename))
                    SourceData = SourceData.loc[:, ~SourceData.columns.str.contains('^Unnamed')]
                    SourceData = [SourceData[a] for a in SourceData.columns if 'FSy' in a]
                    GetT_FDomainGraph(filename, SourceData, filepath, p)
                     
                elif os.path.splitext(filename)[1] == '.mat':
                    data = scipy.io.loadmat(os.path.join(filepath, filename))
                    SourceData = pd.DataFrame()
                    var_names = data.keys()
                    fe_vars = [var for var in var_names if 'FE' in var]
                    
                    fe_data = {var: data[var] for var in fe_vars}
                    # SourceData[fe_vars] = pd.Series(fe_data)
                    for key in fe_vars:
                        a=fe_data[key].flatten()
                        SourceData[key] = pd.Series(a)
                        SourceData = [SourceData[a] for a in SourceData.columns if 'FE' in a]
                        GetT_FDomainGraph(filename, SourceData, filepath, p)
                else :
                    continue
                print(f'{filepath}打印完成')
            except Exception as e:
                print(e)
                print(e.args)

if __name__ == '__main__':
    # path = r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data\with_box\1500\0'
    # main(path)
    # path = r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data\with_box\2000\0'
    # main(path)
    # path = r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data\with_box\2500\0'
    # main(path)
    # path = r'E:\毕设论文\轴承数据集\轴承数据集\StandardSamples\data\with_box\3000\0'
    # main(path)
    # path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1730\7\outer_ring_3'
    # main(path)
    # path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1730\21'
    # main(path)
    # path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1772\21'
    # main(path)
    path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1750\21'
    main(path)
    # path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1797\21'
    # main(path)