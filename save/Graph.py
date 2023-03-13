import os
import pandas as pd
import math
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import pywt
import multiprocessing as mp
# from DataRename import FileRec


def All_GetTimeDomainStatistics(file_path):
    # All系列是文件夹下的所有.csv文件批处理的程序
    # All_GetTimeDomainStatistics 输出时域数据故障统计特征
    Time_stat = pd.DataFrame([], columns=['Filename', 'RMS', 'Peak', 'Crest', 'Skewness', 'Kurtosis'])
    for filepath, dirnames, filenames in os.walk(file_path):
        # 每个文件的时域频域图
        for filename in filenames:

            if os.path.splitext(filename)[1] != '.csv':
                break

            SourceData = pd.read_csv(os.path.join(filepath, filename))
            SourceData = SourceData.loc[:, ~SourceData.columns.str.contains('^Unnamed')]
            db = SourceData.values
            ave = db.mean(axis=0)
            kurt = []; peak = []; crest = []; skew = []; rms = []
            for j in range(db.shape[1]):
                n = db.shape[0]
                a = db[:, j]
                Sum_minsqr = 0
                Sum_sqr = 0
                Sum_min3 = 0
                for i in a:
                    Sum_minsqr += (i - ave[j]) ** 2
                    Sum_min3 += (i - ave[j]) ** 3
                    Sum_sqr += i ** 2
                kurt.append((n * Sum_minsqr)/(Sum_minsqr ** 2))
                rms.append(math.sqrt(Sum_sqr/n))
                skew.append((Sum_min3/n)/math.sqrt(Sum_minsqr))
                peak.append(max(a))
                crest.append(max(a)/math.sqrt(Sum_sqr/n))
            ts = pd.DataFrame({'Filename': filename, 'RMS': [rms], 'Peak': [peak], 'Crest': [crest], 'Skewness': [skew],
                               'Kurtosis': [kurt]})
            Time_stat = pd.concat([Time_stat, ts])

    return Time_stat

def All_GetFreqDomaingraph(data_path, save_path):

    for filepath, dirnames, filenames in os.walk(data_path):
        # 每个文件的时域频域图
        for filename in filenames:
            if os.path.splitext(filename)[1] != '.csv':
                break
            SourceData = pd.read_csv(os.path.join(filepath, filename), header=None, names=['ch1', 'ch2', 'ch3', 'ch4',
                                                                                           'ch5', 'ch6', 'ch7'])
            SourceData.reset_index(drop=True, inplace=True)
            SourceData = SourceData.drop([0, 1, 2, 3])
            SourceData = SourceData.dropna(axis=0, how='any')  # 丢弃空值行
            fft_data=[]
            ps_data=[]
            ceps_data=[]
            for db in SourceData.columns:
                chi = SourceData[db].to_numpy()
                # 获取fft数值
                fft_y = fft(chi)
                N = 12*1024
                abs_y = np.abs(fft_y)
                normal_y = abs_y/N
                half_y = normal_y[range(int(N/2))]
                fft_data.append(half_y)

                # 获取power spectrum功率谱值
                # 自相关法
                chi = chi.astype(np.float64)
                cor_x = np.correlate(chi, chi, 'same')
                cor_x = fft(cor_x, N)
                ps_cor = np.abs(cor_x)
                ps_cor = ps_cor/np.max(ps_cor)
                ps_data.append(20*np.log10(ps_cor[:N//2]))

                # 获取 Cepstrum 倒谱值
                ceps = np.fft.ifft(np.log(fft_y)).real
                ceps_data.append(np.log(np.abs(ceps))[:N//2])


            fft_data = pd.DataFrame(fft_data, columns=None, index=['fft1', 'fft2', 'fft3', 'fft4', 'fft5', 'fft6',
                                                                   'fft7'])
            ps_data = pd.DataFrame(ps_data, columns=None, index=['ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7'])
            ceps_data = pd.DataFrame(ceps_data, columns=None, index=['ceps1', 'ceps2', 'ceps3', 'ceps4', 'ceps5', 'ceps6',
                                                                   'ceps7'])
            fft_data = fft_data.T
            ps_data = ps_data.T
            ceps_data = ceps_data.T
            graph_data = SourceData[4000:10000].reset_index(drop=True)
            graph_data = graph_data.join([fft_data, ps_data, ceps_data])
            graph_data.plot(subplots=True, layout=(4, 12), figsize=(72, 24), sharex=False)
            filename = os.path.splitext(filename)[0]
            plt.savefig(os.path.join(save_path, filename.replace(".", "、")))
            plt.clf()
            plt.close()

def All_GetT_FDomainGraph(data_path,save_path):

    p = mp.Pool(processes=48)
    # 并行CPU数
    for filepath, dirnames, filenames in os.walk(data_path):
        # 每个文件的时域频域图
        for filename in filenames:
            if os.path.splitext(filename)[1] != '.csv':
                break
            SourceData = pd.read_csv(os.path.join(filepath, filename), header=None,
                                     names=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7'])
            SourceData.reset_index(drop=True, inplace=True)
            SourceData = SourceData.drop([0])
            SourceData = SourceData.dropna(axis=0, how='any')  # 丢弃空值行

            # 尝试使用多进程计算CWT
            df = []
            df = [SourceData[a] for a in SourceData.columns]
            df_t = [(df[a][0:3096], np.arange(1, 1024), 'morl') for a in range(7)]
            result = p.starmap(pywt.cwt, df_t)
            fig = plt.figure(figsize=[6, 24], dpi=300)
            for a in range(7):
                plt.subplot(7, 1, a+1)
                plt.imshow(result[a][0])
                plt.set_cmap('hot')
                plt.colorbar(shrink=0.5)

            plt.savefig(os.path.join(save_path, filename.replace(".", "、")))
            plt.close(fig)

def GetTimeDomainStatistics(filename, df):

    db = df.values
    ave = db.mean(axis=0)
    kurt = []; peak = []; crest = []; skew = []; rms = []
    for j in range(db.shape[1]):
        n = db.shape[0]
        a = db[:, j]
        Sum_minsqr = 0
        Sum_sqr = 0
        Sum_mean3 = 0
        Sum_mean4 = 0
        for i in a:
            Sum_sqr = i**2
            Sum_minsqr += (i - ave[j]) ** 2
            Sum_mean3 += (i - ave[j]) ** 3
            Sum_mean4 += (i - ave[j]) ** 4
            Sum_sqr += i ** 2
        kurt.append((n * Sum_mean4)/(Sum_minsqr ** 2))
        rms.append(math.sqrt(Sum_sqr/n))
        skew.append((Sum_mean3/n)/math.sqrt(Sum_minsqr))
        peak.append(max(a))
        crest.append(max(a)/math.sqrt(Sum_sqr/n))

    ts = pd.DataFrame({'Filename': filename, 'RMS': [rms], 'Peak': [peak], 'Crest': [crest], 'Skewness': [skew],
                       'Kurtosis': [kurt]})

    return ts

def GetFreqDomaingraph(name, df, path):

    fft_data=[]
    ps_data=[]
    ceps_data=[]
    for column in df.columns:
        chi = df[column].to_numpy()
        # 获取fft数值
        fft_y = fft(chi)
        N = 12*1024
        abs_y = np.abs(fft_y)
        normal_y = abs_y/N
        half_y = normal_y[range(int(N/2))]
        fft_data.append(half_y)

        # 获取power spectrum功率谱值
        # 自相关法
        cor_x = np.correlate(chi, chi, 'same')
        cor_x = fft(cor_x, N)
        ps_cor = np.abs(cor_x)
        ps_cor = ps_cor/np.max(ps_cor)
        ps_data.append(20*np.log10(ps_cor[:N//2]))

        # 获取 Cepstrum 倒谱值
        ceps = np.fft.ifft(np.log(fft_y)).real
        ceps_data.append(np.log(np.abs(ceps))[:N//2])

    fft_data = pd.DataFrame(fft_data, columns=None, index=['fft1', 'fft2', 'fft3', 'fft4', 'fft5', 'fft6', 'fft7'])
    ps_data = pd.DataFrame(ps_data, columns=None, index=['ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7'])
    ceps_data = pd.DataFrame(ceps_data, columns=None, index=['ceps1', 'ceps2', 'ceps3', 'ceps4', 'ceps5', 'ceps6', 'ceps7'])
    fft_data = fft_data.T
    ps_data = ps_data.T
    ceps_data = ceps_data.T
    graph_data = df[4000:10000].reset_index(drop=True)
    graph_data = graph_data.join([fft_data, ps_data, ceps_data])
    graph_data.plot(subplots=True, layout=(4, 7), figsize=(42, 24), sharex=False)
    filename = os.path.splitext(name)[0]+'.png'
    plt.savefig(os.path.join(path, filename))
    plt.clf()
    plt.close()


def GetT_FDomainGraph(name, df, path, p):

    filename = os.path.splitext(name)[0]+'.png'
    df = [df[a] for a in df.columns]
    df_t = [(df[a][0:3096], np.arange(1, 1024), 'morl') for a in range(7)]
    result = p.starmap(pywt.cwt, df_t)
    fig = plt.figure(figsize=[6, 24], dpi=300)
    for a in range(7):
        plt.subplot(7, 1, a+1)
        plt.imshow(result[a][0])
        plt.set_cmap('hot')
        plt.colorbar(shrink=0.5)

    plt.savefig(os.path.join(path, filename))
    plt.clf()
    plt.close(fig)


def GetTimeDomainGraph(filename, data, save_path):
    rms = data.at[0, 'RMS']
    peak = data.at[0, 'Peak']
    crest = data.at[0, 'Crest']
    skew = data.at[0, 'Skewness']
    kurt = data.at[0, 'Kurtosis']
    stat_plot_y = [rms, peak, crest, skew, kurt]
    stat_plot_x = ['ch'+str(a+1) for a in range(6)]
    plt.figure(figsize=[6.4, 6.4*2.5])
    for a in range(5):
        plt.subplot(51*10+a+1)
        plt.bar(stat_plot_x, stat_plot_y[a][0:6])
        plt.xlabel(data.columns[a+1])
    filename = os.path.splitext(filename)[0]+'.png'
    plt.savefig(os.path.join(save_path, filename))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    # 此处所有处理 使用的均是标准数据集中的内容

    # %%%%%%%%%%%%%%%% 针对所有数据批处理 %%%%%%%%%%%%%%%%%%%%%%%%%%%
    path = r'D:\python_workfile\ML_Classify\samples\data\vary_condition_multi_load_1000'
    save_path_t = r'D:\python_workfile\ML_Classify\graph'
    save_path_f = r'D:\python_workfile\ML_Classify\graph'
    save_path_tf = r'D:\python_workfile\ML_Classify\graph'
    # # graph_path = r'D:\python_workfile\DataManage\F_Picture'
    # # tf_save_path = r'D:\python_workfile\DataManage\TF_Picture'
    # # Time = All_GetTimeDomainStatistics(path)
    # All_GetFreqDomaingraph(path, save_path_f)
    # All_GetT_FDomainGraph(path, save_path_tf)

    #
    #  %%%%%%%%%%%%%%%%% 针对某类批处理 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 根据 故障类型/载荷/转速 筛选数据文件
    fault = 'baseline'
    speed = ''
    load = ''
    word_list = [fault, speed, load]
    time_stat_namelist = ['']
    time_stat_chanlist = [1, 2, 3, 4, 5, 6, 7]
    time_stat = pd.DataFrame()

    p = mp.Pool(processes=12)

    for filepath, dirnames, filenames in os.walk(path):
        # 获取每个数据文件的时域特征值、频谱图、时频图
        for filename in filenames:

            if os.path.splitext(filename)[1] != '.csv':
                break
            if all(word in filename for word in word_list):
                SourceData = pd.read_csv(os.path.join(filepath, filename))
                SourceData = SourceData.loc[:, ~SourceData.columns.str.contains('^Unnamed')]
                if time_stat.empty == 1:
                    stat = GetTimeDomainStatistics(filename, SourceData)
                    time_stat = GetTimeDomainStatistics(filename, SourceData)
                else:
                    stat = GetTimeDomainStatistics(filename, SourceData)
                    time_stat = pd.concat([time_stat, stat], axis=0)

                GetTimeDomainGraph(filename, stat, save_path_t)
                GetFreqDomaingraph(filename, SourceData, save_path_f)
                # GetT_FDomainGraph(filename, SourceData, save_path_tf, p)


    # # 多个数据文件的对比
    # FileRec(r'D:/DATABASE/ZXJ_GD', r'D:/DATABASE/ZXJ_GD/data_index.xls', r'D:/DATABASE/ZXJ_GD/standard')
