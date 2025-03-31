"""
    基于排行榜的推荐
    基于协同过滤的推荐
    基于矩阵分解的推荐
    基于GBDT+LR预估的推荐
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from surprise import KNNBasic
from surprise import SVD
from surprise import Reader,Dataset,accuracy
from surprise.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv(r'C:\opencv\Recommend_System\data\train_data.csv',
                   sep='\t',header=None,names=['user','song','play_count'],nrows=2000000)
data.head()

# 查看数据内存信息
data.info()

# label编码
user_encoder = LabelEncoder()
data['user'] = user_encoder.fit_transform(data['user'].values)

song_encoder = LabelEncoder()
data['song'] = song_encoder.fit_transform(data['song'].values)

# 数据类型转换
data.astype({'user': 'int32', 'song': 'int32','play_count': 'int32'})

# 当前内存情况
data.info()

# 用户的歌曲播放总量的分布
user_playcounts = {}
for user,group in data.groupby('user'):
    user_playcounts[user] = group['play_count'].sum()

# 作图
sns.distplot(list(user_playcounts.values()),bins=5000,kde=False)
plt.xlim(0,200)
plt.xlabel('play_count')
plt.ylabel('nums of user')
plt.show()

temp_user = [user for user in user_playcounts.keys() if user_playcounts[user] > 100]
temp_playcounts = [playcounts for user,playcounts in user_playcounts.items() if playcounts > 100]

print('歌曲播放量大于100的用户数量占总体用户数量的比例为', str(round(len(temp_user)/len(user_playcounts), 4)*100)+'%')
print('歌曲播放量大于100的用户产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(user_playcounts.values())*100, 4))+'%')
print('歌曲播放量大于100的用户产生的数据占总体数据的比例为', str(round(len(data[data.user.isin(temp_user)])/len(data)*100, 4))+"%")

# 过滤掉歌曲播放量少于100的用户数据
data = data[data.user.isin(temp_user)]

#  song_playcounts字典，记录每首歌的播放量
song_playcounts = {}
for song,group in data.groupby('song'):
    song_playcounts[song] = group['play_count'].sum()

# 作图
sns.distplot(list(song_playcounts.values()),bins=10000,kde=False)
plt.xlim(0,100)
plt.xlabel('play_count')
plt.ylabel('nums of song')
plt.show()

temp_song = [song for song in song_playcounts.keys() if song_playcounts[song] > 50]
temp_playcounts = [playcounts for song,playcounts in song_playcounts.items() if playcounts > 50]

print('播放量大于20的歌曲数量占总体歌曲数量的比例为', str(round(len(temp_song)/len(song_playcounts), 4)*100)+'%')
print('播放量大于20的歌曲产生的播放总量占总体播放总量的比例为', str(round(sum(temp_playcounts) / sum(song_playcounts.values())*100, 4))+'%')
print('播放量大于20的歌曲产生的数据占总体数据的比例为', str(round(len(data[data.song.isin(temp_song)])/len(data)*100, 4))+"%")

# 过滤掉播放量小于50的歌曲
data = data[data.song.isin(temp_song)]

# 对db文件的处理
conn = sqlite3.connect(r'C:\opencv\Recommend_System\data\train_data.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
cur.fetchall()

# 获取数据的dataFrame
track_metadata_df = pd.read_sql(con=conn,sql='selct * from songs')

# 对于之前的歌曲编码，我们给出一个字典，对歌曲和编码进行一一映射
song_labels = dict(zip(song_encoder.classes_,range(len(song_encoder.classes_))))

# 对于那些在之前没有出现过的歌曲，我们直接给出一个最大的编码
encoder = lambda x: song_labels[x] if x in song_labels.keys() else len(song_labels)

# 对数据进行labelencoder
track_metadata_df['song_id'] = track_metadata_df['song_id'].apply(encoder)

# 对song_id重命名为song
track_metadata_df['user_id'] = track_metadata_df.rename(columns={'song_id':'song'})

# 根据特征song进行拼接，将拼接后的数据重新命名为data
data = pd.merge(data,track_metadata_df,on='song')

# 为了降低内存，同样进行内存转换
data.info()
data.columns
data = data.astype({'play_count': 'int32', 'duration': 'float32', 'artist_familiarity': 'float32',
            'artist_hotttnesss': 'float32', 'year': 'int32', 'track_7digitalid': 'int32'})
print(' ')
data.info()

# 去重
data.drop_duplicates(inplace=True)
# 丢掉无用信息
data.drop(['track_id', 'artist_id', 'artist_mbid', 'duration', 'track_7digitalid', 'shs_perf', 'shs_work'], axis=1, inplace=True)

data.info()
data.head()

# 字典artist_playcounts记录每个歌手获得的点击量
artist_playcounts = {}
for artist,group in data.groupby('artist_name'):
    artist_playcounts[artist] = group['play_count'].sum()

# 作图
plt.figure(figsize=(12,8))
wc = WordCloud(width=1000,height=800)
wc.generate_from_frequencies(artist_playcounts)
plt.imshow(wc)
plt.axis('off')
plt.show()

# 字典release_playcounts记录每个专辑获得的点击量
release_playcounts = {}
for release, group in data.groupby('release'):
    release_playcounts[release] = group['play_count'].sum()

# 作图
plt.figure(figsize=(12, 8))
wc = WordCloud(width=1000, height=800)
wc.generate_from_frequencies(release_playcounts)
plt.imshow(wc)
plt.axis('off')
plt.show()

# 字典song_playcounts记录每首歌获得的点击量
song_playcounts = {}
for song, group in data.groupby('title'):
    song_playcounts[song] = group['play_count'].sum()

# 作图
plt.figure(figsize=(12, 8))
wc = WordCloud(width=1000, height=800)
wc.generate_from_frequencies(song_playcounts)
plt.imshow(wc)
plt.axis('off')
plt.show()

# 基于排行榜进行推荐
def recommendation_basedonPopularity(df,N=5):
    my_df = df.copy()
    # 字典song_peopleplay,记录每首歌听过的人数
    song_peopleplay = {}
    for song,group in my_df.groupby('title'):
        song_peopleplay[song] = group['user'].count()

    # 根据人数从大到小排序，并推荐前N首歌
    sorted_dict = sorted(song_peopleplay.items(), key=lambda x : x[1], reverse=True)[:N]
    # 取出歌曲
    return list(dict(sorted_dict).keys())

# 测试推荐结果
recommendation_basedonPopularity(data,N=5)

(data['play_count'].min(),data['play_count'].max())

# 每个用户点击量的平均数
user_averageScore = {}
for user,group in data.groupby('user'):
    user_averageScore[user] = group['play_count'].mean()

data['rating'] = data.play(lambda x: np.log(2 + x.play_count / user_averageScore[x.user]),axis=1)

sns.distplot(data['rating'].values,bins=100)
plt.show()


# 得到用户-音乐评分矩阵
user_item_rating = data[['user', 'song', 'rating']]
user_item_rating = user_item_rating.rename(columns={'song': 'item'})

# 做itemCF推荐

# 阅读器
reader = Reader(line_format='user item rating',sep=',')
# 载入数据
raw_data = Dataset.load_from_df(user_item_rating, reader=reader)
# 分割数据集
kf = KFold(n_splits=5)
# 构建模型
knn_itemcf = KNNBasic(k=40,sim_options={'user_based':False})
# 训练数据集，返回rmse误差
for trainset,testset in kf.split(raw_data):
    knn_itemcf.fit(trainset)
    predictions = knn_itemcf.predict(raw_data[testset])
    accuracy.rmse(predictions,verbose=True)

# 用户听过的歌曲合集
user_songs = {}
for user,group in user_item_rating.groupby('user'):
    user_songs[user] = group['item'].values.tolist()

# 歌曲集合
songs = user_item_rating['item'].unique().tolist()

# 歌曲id和歌曲名称对应关系
songID_titles = {}
for index in data.index:
    songID_titles[data.loc[index,'song']] = data.loc[index,'title']

user_item_rating.head()

# itemCF推荐
def recommendation_basedonItemCF(userID,N=5):
    # 用户听过的音乐列表
    used_items = user_songs[userID]

    # 用户对未听过的音乐的评分
    item_ratings = {}
    for item in songs:
        if item not in used_items:
            item_ratings[item] = knn_itemcf.predict(userID,item).est

    # 找出评分靠前的5首歌曲
    song_ids = dict(sorted(item_ratings.items(), key=lambda x : x[1], reverse=True)[:N])
    song_topN = [songID_titles[s] for s in song_ids.keys()]

    return song_topN

recommendation_basedonItemCF(29990)

# userCF推荐

# 阅读器
reader = Reader(line_format='user item rating',sep=',')
# 载入数据
raw_data = Dataset.load_from_df(user_item_rating, reader=reader)
# 分割数据集
kf = KFold(n_splits=5)
# 构建模型
knn_usercf = KNNBasic(k=40,sim_options={'user_based':True})
# 训练数据集，并返回rmse误差
for trainset,testset in kf.split(raw_data):
    knn_usercf.fit(trainset)
    predcitions = knn_usercf.test(testset)
    accuracy.rmse(predcitions,verbose=True)

def recommendation_basedonUserCF(userID,N=5):
    # 用户听过的音乐列表
    used_items = user_songs[userID]

    # 用户对未听过的音乐的评分
    item_ratings = {}
    for item in songs:
        if item not in used_items:
            item_ratings[item] = knn_usercf.predict(userID,item).est

    # 找出评分靠前的5首歌曲
    song_ids = dict(sorted(item_ratings.items(), key=lambda x : x[1], reverse=True)[:N])
    song_topN = [songID_titles[s] for s in song_ids.keys()]

    return song_topN

recommendation_basedonUserCF(29990)

# 矩阵分解（SVD）

# 阅读器
reader = Reader(line_format='user item rating', sep=',')
# 载入数据
raw_data = Dataset.load_from_df(user_item_rating, reader=reader)
# 分割数据集
kf = KFold(n_splits=5)
# 构建模型
algo = SVD(n_factors=40, biased=True)
# 训练数据集，并返回rmse误差
for trainset, testset in kf.split(raw_data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)

    # 矩阵分解 推荐

def recommendation_basedonMF(userID, N=5):
    # 用户听过的音乐列表
    used_items = user_songs[userID]

    # 用户对未听过音乐的评分
    item_ratings = {}
    for item in songs:
        if item not in used_items:
            item_ratings[item] = algo.predict(userID, item).est

    # 找出评分靠前的5首歌曲
    song_ids = dict(sorted(item_ratings.items(), key=lambda x: x[1], reverse=True)[:N])
    song_topN = song_ids

    return song_topN

recommendation_basedonMF(29990)

# 复制原data数据
rank_data = data.copy()
# 去掉无用的title列
rank_data.drop('title',axis=1,inplace=True)

# 将object类型数据用labelencoder编码
release_encoder = LabelEncoder()
rank_data['release'] = release_encoder.fit_transform(rank_data['release'].values)

artist_name_encoder = LabelEncoder()
rank_data['artist_name'] = artist_name_encoder.fit_transform(rank_data['artist_name'].values)

# 根据rating的取值，更新rating值
rank_data['rating'] = rank_data['rating'].apply(lambda x: 0 if x < 0.7 else 1)

rank_data.head()

# 取出20%的数据作为数据集
small_data = rank_data.sample(frac=0.2)
# 将数据集分成gbdt训练集和lr训练集
X_gbdt,X_lr,y_gbdt,y_lr = train_test_split(small_data.iloc[:,:-1].values,small_data.iloc[:,-1].values,test_size=0.5)

depth = 3
n_estimator = 50

print('当前n_estimators=',n_estimator)
# 训练gbdt
gbdt = GradientBoostingClassifier(n_estimators=n_estimator, max_depth=depth, min_samples_split=3, min_samples_leaf=2)
gbdt.fit(X_gbdt, y_gbdt)

print('当前gbdt训练完成！')

# one-hot编码
onehot = OneHotEncoder()
onehot.fit(gbdt.apply(X_gbdt).reshape(-1,n_estimator))

# 对gbdt结果进行one-hot编码，然后训练lr
lr = LogisticRegression()
lr.fit(onehot.transform(gbdt.apply(X_gbdt)).reshape(-1,n_estimator), y_gbdt)

print('当前lr训练完成！')

# 用auc作为指标
lr_pred = lr.predict(onehot.transform(gbdt.apply(X_lr).reshape(-1, n_estimator)))
auc_score = roc_auc_score(y_lr, lr_pred)

print('当前n_estimators和auc分别为', n_estimator, auc_score)
print('#'*40)

# 推荐
def recommendation(userID):
    # 召回50首歌
    recall = recommendation_basedonMF(userID,50)

    print('召回完毕！')

    # 根据召回的歌曲信息，写出特征向量
    feature_lines = []
    for song in recall.keys():
        feature = rank_data[rank_data.song == song].values[0]
        #  除去其中的rating，将user数值改成当前userID
        feature = feature[:-1]
        feature[0] = userID
        # 放入特征向量
        feature_lines.append(feature)

    # 用gbdt+lr计算权重
    weights = lr.predict_proba(onehot.transform(gbdt.apply(feature_lines).reshape(-1,n_estimator)))[:,1]

    # print(wight.shape)
    print('排序权重计算完毕！')

    # 计算最终得分
    score = {}
    i = 0
    for song in recall.keys():
        score[song] = recall[song] * weights[i]
        i += 1

    # print(score)

    # 选出排名前5的歌曲id
    song_ids = dict(sorted(score.items(),key=lambda x : x[1], reverse=True)[:5])
    # 前5歌曲名称
    song_topN = [songID_titles[s] for s in song_ids.keys()]

    print('最终推荐列表为')

    return song_topN

# 测试
recommendation(29990)