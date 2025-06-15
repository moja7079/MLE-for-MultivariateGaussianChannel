import numpy as np
import cupy as cp
import itertools
import time
import math
import sys

# cp.set_printoptions(precision=100000, suppress=True)
# setting--------------------------------
delta=0.1
t2=0.03
snrdB_iteration=100
snrdB_default=-4
error_iteration=100
limited_memory=2**18
n = 24  # 次元数
k = 12
G = cp.array([
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
                            ])
#setting_end-------------------------------

def main():



    with open('data/mle.txt', 'w') as f:
        print(f"初期設定-------------------------",file=f)
        print(f"符号長n:{n}",file=f)
        print(f"情報記号数:{k}",file=f)
        print(f"delta:{delta}",file=f)
        print(f"t2:{t2}",file=f)
        print(f"生成行列G\n{G}",file=f)
        # print(f"情報記号:\n{m}",file=f)
        # print(f"送信符号語:\n{x}",file=f)
        print(f"情報記号はランダム生成",file=f)
        print(f"snrdB_iteration:{snrdB_iteration}",file=f)
        print(f"snrdb_default:{snrdB_default}",file=f)
        print(f"error_iteration:{error_iteration}",file=f)
        print(f"初期設定end-----------------------",file=f)


    with open("data/mle_onlydata.txt", "w") as f2:
        print(f"初期設定-------------------------",file=f2)
        print(f"符号長n:{n}",file=f2)
        print(f"情報記号数:{k}",file=f2)
        print(f"delta:{delta}",file=f2)
        print(f"t2:{t2}",file=f2)
        print(f"生成行列G\n{G}",file=f2)
        # print(f"情報記号:\n{m}",file=f2)
        # print(f"送信符号語:\n{x}",file=f2)
        print(f"情報記号はランダム生成",file=f2)
        print(f"snrdB_iteration:{snrdB_iteration}",file=f2)
        print(f"snrdb_default:{snrdB_default}",file=f2)
        print(f"error_iteration:{error_iteration}",file=f2)
        print(f"初期設定end-----------------------",file=f2)

        pass


    for i in range(snrdB_iteration):
        snrdB=i+snrdB_default
        t1 = create_t_1_from_snrdB(snrdB, n, k)

        #相関通信路
        sigma = create_sigma(n, t1)
        # sigma=sigma+1e-5*cp.eye(n)
        # print(f"sigma:\n{sigma}")
        #相関通信路end

        #無相関通信路
        # sigma=t1*cp.eye(n) #単位行列
        #無相関通信路end

        # min_eig = cp.min(cp.real(cp.linalg.eigvalsh(sigma)))
        # if min_eig < 0:
        #     sigma -= 10*min_eig * cp.eye(*sigma.shape)

            
        word_error_count = 0
        iteration_count = 0
        while word_error_count < error_iteration:
            time_start1 = time.time()
            # G = generator_matrix_random(n, k)
            m = m_create(k)
            x = codeword_create(G, m)
            # v_bpsk = firstonly_codeword_create(G, m)
            # x=cp.array([1 , 0 ,0 , 0])
            r = received_sequence_create(x, sigma)
            # print(f"r:\n{r}")
            # r=cp.array([0.07529169 ,0.8168622  ,1.34055932 ,1.22124756])
            # x=cp.array([1 ,1 ,1 ,0])
            # r=cp.array([-0.56528307, -0.30235741,  0.14088456, -1.86911216])
            # G=cp.array([[1, 0, 1, 0, 0],
            #             [1, 0, 0, 1, 1],
            #             [1, 1, 1, 1, 0]])
            # m=cp.array([1,1,0])
            # v_bpsk = codeword_create(G, cp.array([m]))
            # r=cp.array([46469.37048167, 62456.81929849, 73647.04221187, 76493.30585836,
            #             68700.04019895])
            

            # correct_loglikehood = logpdf(r, v_bpsk, sigma)
            print("----------------------------------")
            print(f"snrdB:{snrdB}")
            print(f"情報記号:{m}")
            print(f"送信符号語:{x}")
            # print(f"正解の尤度:{correct_loglikehood}")
            # print(f"G:\n{G}")

            max_loglikehood, x_estimate=batch_mle_calculate(G,k,r,sigma,limited_memory)
            # max_loglikehood, x_estimate=pre_batch_max_loglikehood_estimate_calculate(G, k, r, sigma)
            # print(f"v_bpsk:\n{v_bpsk}")
            # print(f"estimate_v:\n{x_estimate}")
            if cp.all(x == x_estimate):
                print(f"復号成功")
            else:
                print(f"復号失敗")
                # with open("data/decoding_failed.txt", "a") as f3:
                #     print(f"G:\n{G}",file=f3)
                    
                word_error_count +=1

            time_end1 = time.time()
            print(f"出力尤度:{max_loglikehood}")
            print(f"推定符号語:{x_estimate}")
            print(f"現在までの復号誤り個数:{word_error_count}")
            print(f"現在までの反復回数:{iteration_count}")
            print(f"時間:{time_end1-time_start1}")

            iteration_count +=1

        print(f"wer:{word_error_count/iteration_count}")
        with open("data/mle.txt","a") as f:
            print(f"------------------------------",file=f)
            print(f"snrdB:{snrdB}",file=f)
            print(f"{iteration_count}回目,合計誤り回数:{word_error_count}",file=f)
            print(f"WER:{word_error_count/iteration_count}",file=f)
        with open("data/mle_onlydata.txt","a")as f2:
            print(f"{word_error_count/iteration_count},",file=f2)

    # ------------------------------------------

    return 0

def generator_matrix_random(n, k):
    G = cp.random.randint(0, 2, size=(k,n))
    return G


def m_create(k):
    return cp.random.randint(0, 2, size=(k,))


def codeword_create(G, m):
    return cp.dot(m, G) % 2


def create_t_1_from_snrdB(snrdB, n, k):
    Eb = n/k  # =1/R
    N0 = Eb/(cp.power(10, snrdB/10))
    t1 = N0/2  # 分散はN0/2らしい
    print(f"Eb:{Eb}")
    print(f"N0:{N0}")
    print(f"t1(分散):{t1}")
    return t1

def variance_generate(snrdB,n,k):
    R=k/n
    snr=10**(snrdB/10)
    variance=1/(2*R*snr)
    return variance

def noise_create(sigma):
    n = sigma.shape[0]
    mu = cp.zeros(n)
    # rng = np.random.default_rng()
    # noises = rng.multivariate_normal(mu, sigma)
    noises = cp.random.multivariate_normal(mu, sigma, method='cholesky',dtype=np.float64,check_valid='warn')
    # noises = cp.random.multivariate_normal(mu, sigma, method='svd',dtype=np.float64,check_valid='raise',tol=1e-08)
    # noises = cp.random.multivariate_normal(mu, sigma, method='cholesky',dtype=np.float64,check_valid='raise',tol=1e-08)
    # noises=multivariate_normal.rvs(mean=None,cov=sigma,size=1,random_state=None)
    
    # print(f"shape_noise_create:{n}")
    # print(f"mu:{mu}")
    # print(f"noise:\n{noises}")

    return noises


def received_sequence_create(x, sigma):
    x_bpsk=cp.where(x==0,1,-1)
    z = noise_create(sigma)
    # z=cp.array([1.2,0.8,1,0.7,0.5])
    # z=cp.array([-2.3,-1.9,-2.0,-2.4,-2.6])
    # z=np.array(z)
    r = x_bpsk+z
    # print(f"v:\n{v}")
    # print(f"noises:\n{z}")
    # print(f"r:\n{r}")
    return r


def k_ij(i, j, t1):  # カーネル
    # setting------------
    # t_1 = t1  # たぶんt_1=N_0なので、snr=1/N_0
    # delta = 0.1  # 0.1
    # t_2 = 0.5  # 0.5- 4 相関の大きさ
    # -------------------
    return t1*cp.exp((-cp.power(delta*i-delta*j, 2))/t2)


def create_sigma(n, t1):
    sigma = cp.empty((n, n))
    for i in range(n):
        for j in range(n):
            sigma[i, j] = k_ij(i, j, t1)
    return sigma

def logpdf(x, mean, cov):
    # Cholesky分解を用いて共分散行列を分解

    # L = np.linalg.cholesky(cov)
    # print("ここで鰓ー１")
    # L=cholesky(cov,lower=True) #t2=4,n=13,k=4正定値じゃないエラーでる。ー＞対角成分に微小値解決sigma=sigma+1e-8*np.eye(n)
    # print("ここでえあｒ－２")
    L = cp.linalg.cholesky(cov)
    
    
    # Mahalanobis距離を計算するための補助変数
    dev = x - mean
    # Lの逆行列を求め、devを変換
    inv_dev = cp.linalg.solve(L, dev)  # L^Tに対する線形方程式を解く
    
    # Mahalanobis距離を計算
    maha = cp.dot(inv_dev, inv_dev)  # inv_devの内積
    # print(f"L_inv\n{np.linalg.inv(L)}")
    # print(f"L:\n{L}")
    # print(f"L:\n{L_np}")
    # print(f"dev:\n{dev.T}")
    # print(f"inv_dev:\n{inv_dev}")
    # print(f"maha:\n{maha}")
    # 最終的な対数確率密度関数の値を返す
    return maha


def multi_logpdf(x, means, cov):
    # Cholesky分解を用いて共分散行列を分解
    # L = cholesky(cov, lower=True)  # Lを得る
    L = cp.linalg.cholesky(cov)
    # L=np.linalg.cholesky
    
    # Mahalanobis距離を計算するための補助変数
    # xは1つのベクトル、meansは複数の平均ベクトル
    # 偏差を計算 (k x n) で、各meanとの偏差を計算
    k=means.shape[0]
    n=means.shape[1]
    dev = x-means  # (k x n)
    dev=dev.reshape(k,n,1) #(2^k, n , 1) k=2^kわかりずらい！！！！
    
    # Lの逆行列を求め、devを変換
    # z = np.linalg.solve(L.T, dev.T).T  # L^Tに対する線形方程式を解く
    L_expanded = cp.tile(L, (k, 1, 1))
    # print(f"dev_reshape:\n{dev.shape}")
    # print(f"L:\n{L}")
    # prinxt(f"L_expand:\n{L_expanded}")
    # print("ここまでおｋ１")
    z = cp.linalg.solve(L_expanded, dev)
    # print("ここまでおｋ２")
    # z = np.linalg.solve(L, dev)
    # z = cp.einsum('ij,kjl->kil', cp.linalg.inv(L), dev)#inv使ってるやんけ
    # Mahalanobis距離を計算
    # maha = np.dot(z.transpose(0, 2, 1), z)  # (k x k) の距離行列を得る
    maha=z.transpose(0, 2, 1)@z
    # 各meanに対するマハラノビス距離の負の値を返す
    # print(f"L:\n{L}")
    # print(f"dev:\n{dev}")
    # print(f"z:\n{z}")
    # print(f"maha:\n{maha}")
    
    # print(f"L:\n{L}")
    # print(f"dev:\n{dev}")
    # print(f"L_expanded:\n{L_expanded}")
    # print(f"z:\n{z}")
    # print(f"maha:\n{maha}")
    return maha  # 各meanに対する対数確率密度を返す



def batch_mle_calculate(G,k,r,sigma,limited_memory):
    values=[0,1]
    total_iterations=math.ceil(2**k/limited_memory)
    max_loglikehood=sys.float_info.max
    codeword_estimate=[]
    print(f"total_iterations:{total_iterations}")

    for batch_count in range(total_iterations):
        # print(f"iteration")
        m_combinations=cp.array(list(itertools.islice(itertools.product(values,repeat=k),limited_memory*batch_count, limited_memory*(batch_count+1))))
        candidate_codewords=codeword_create(G,m_combinations)
        candidate_codeword_bpsks=cp.where(candidate_codewords==0,1,-1)
        loglikehoods=multi_logpdf(r,candidate_codeword_bpsks,sigma)
        # loglikehoods=new_multi_logpdf(r,candidate_codeword_bpsks,sigma)
        # print(f"候補符号語:\n{candidate_codeword_bpsks}")
        # print(f"尤度\n{loglikehoods}")
        pre_max_loglikehood=cp.min(loglikehoods)
        pre_max_loglikehood_index=cp.argmin(loglikehoods)
        pre_codeword_estimate=candidate_codewords[pre_max_loglikehood_index]
        if cp.isnan(pre_max_loglikehood):
            raise ValueError("エラーが発生しました:尤度がすべてnan")
        if max_loglikehood> pre_max_loglikehood:
            max_loglikehood=pre_max_loglikehood
            codeword_estimate=pre_codeword_estimate

    return max_loglikehood, codeword_estimate
        

if __name__ == "__main__":
    main()