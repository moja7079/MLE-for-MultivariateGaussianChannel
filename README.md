# MLE-for-MultivariateGaussianChannel
【ファイル名】
    MLE_forMultivariateGaussianChannel

【プログラム内容】
    多変量ガウス雑音通信路で最尤復号を実行するプログラム

【開発環境】
    numpy 2.0.2
    cupy 13.4.1
    python 3.12.6

【動作環境】
    windows11

【使い方】
    実行するとdataフォルダに2つのファイルが出力される.
    それぞれ実行条件と実行結果が表示される.
    mle.txt:実行条件と実行結果が記録されたファイル
    mle_onlydata.txt:実行結果が抽出しやすくしたファイル

    プログラムコード8行目から32行目のsetting内の変更することで符号等の条件を変更することができる
    以下setting内の変数の説明
    snrdB_iteration: SNR(dB)の反復回数
    snrdB_default:デフォルトのSNR(dB)値
    SNR_INTERVAL: snrdBの増加量
    word_error_iteration: シミュレーションでワード誤り率の統計を取るために繰り返すワード誤り回数
    order:順序統計量復号におけるOrder
    limited_memory:メモリ使用量を制限する変数
    n:符号長
    k:情報記号数
    G:生成行列

【注意事項】
    delta,t2の変更で雑音成分同士の相関を強くしすぎると,尤度の計算結果にnanが表示され結果が得られない.
