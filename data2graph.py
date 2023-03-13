import os
import pandas as pd
import scipy.io
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import pywt
import multiprocessing as mp
# len(df[0])-128
def GetT_FDomainGraph(name, df, path, p):
    for a in range(0,min(len(df[0])-512, 128*512),128):
        df_t = [(df[0][a:a+512], np.arange(1, 512), 'morl')]
        result = p.starmap(pywt.cwt, df_t)
        fig = plt.figure(figsize=[6, 24], dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.imshow(result[0][0])
        plt.set_cmap('hot')
        plt.axis('off')
        # plt.colorbar(shrink=0.5).remove()
        
        save_path = path.replace('CWRUData', 'CWRUData-picture')
        filename = os.path.splitext(name)[0]+f'_{int(a/128)}'+'.png'
        plt.savefig(os.path.join(save_path, filename),bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close(fig)
        
def main(path):
    p = mp.Pool(processes=12)
    for filepath, dirnames, filenames in os.walk(path):

        for filename in filenames:
            try:
                if os.path.splitext(filename)[1] == '.csv':
                    SourceData = pd.read_csv(os.path.join(filepath, filename))
                    SourceData = SourceData.loc[:, ~SourceData.columns.str.contains('^Unnamed')]
                elif os.path.splitext(filename)[1] == '.mat':
                    data = scipy.io.loadmat(os.path.join(filepath, filename))
                    SourceData = pd.DataFrame()
                    for key in data:
                        if not key.startswith('__'):
                            SourceData[key] = pd.Series(data[key].flatten())
                else :
                    continue
                SourceData = [SourceData[a] for a in SourceData.columns if 'FE' in a]
                GetT_FDomainGraph(filename, SourceData, filepath, p)
                print(f'{filepath}打印完成')
            except Exception as e:
                print(e)
                print(e.args)

if __name__ == '__main__':
    path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1730\21'
    main(path)
    path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1797\7'
    main(path)
    path = r'E:\毕设论文\CWRU\CWRU_xjs\CWRUData\12K_Drive_End\1797\21'
    main(path)
