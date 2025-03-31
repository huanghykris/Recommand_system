# 数据预处理
# 文本清洗
# 矩阵分解
# LDA主题模型(无监督模型)
# 构建推荐引擎

import numpy as np
import pandas as pd
import re
import string

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora,models
from gensim.utils import simple_preprocess

from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# 预处理
medium = pd.read_csv('Medium_aggregateData.csv')
medium.head()

medium = medium[medium['language'] == 'en']
meidum = medium[medium['totalClapCount'] >= 25]

# 整理文章对应标签
def findTags(title):
    rows = medium[medium['title'] == title]
    tags = list(rows['tag_name'].values)
    return tags

titles = medium['title'].unique() # 所有文章名字
tag_dict = {'title':[],'tags':[]} # 文章对应标签

for title in titles:
    tag_dict['title'].append(title)
    tag_dict['tags'].append(findTags(title))

tag_df = pd.DataFrame(tag_dict) # 转换成DF

# 去重
medium = medium.drop_duplicates(subset='title',keep='first')

# 将标签加入到原始DF中
medium['allTags'] = medium['title'].apply(addTags)

# 只保留需要的列
keep_cols = ['title','url','allTags','readingTime','author','text']
medium = medium[keep_cols]

# 标题为空的不要了
null_title = medium[medium['title'].isna()].index
medium.drop(index = null_title,inplace = True)

medium.reset_index(drop = True, inplace = True)

# 文本清洗
def clean_text(text):
    # 去掉http开头那些链接
    text = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+',' ',text)
    # 去掉特殊字符之类的
    text = re.sub('\w*\d\w*',' ',text)
    # 去掉标点符号，将所有字符转换成小写的
    text = re.sub('[%s]' % re.escape(string.punctuation),' ',text.lower())
    # 去掉换行符
    text = text.replace('\n',' ')
    return text

medium['text'] = medium['text'].aplly(clean_text)

# 去停用词
stop_list = STOPWORDS.union(set(['data','ai','learing','time']))

def remove_stopwords(text):
    clean_text = []
    for word in text.split(' '):
        if word not in stop_list and (len(word) > 2):
            clean_text.append(word)

    return ' '.join(clean_text)

medium['text'] = medium['text'].apply(remove_stopwords)

# 词干提取
stemmer = PorterStemmer()

def stem_text(text):
    word_list = []
    for word in text.split(' '):
        word_list.append(stemmer.stem(word))

    return ' '.join(word_list)

medium['text'] = medium['text'].apply(stem_text)

medium.to_csv('pre_processed.csv')

# TFIDF处理
vectorizer = TfidfVectorizer(stop_words=stop_list,ngram_range=(1,1))
doc_word = vectorizer.fit_transform(medium['text'])

# SVD矩阵分解
svd = TruncatedSVD(8)
docs_svd = svd.fit_transform(doc_word)

# 打印函数
def display_topics(model,feature_names,no_top_words,no_top_topics,topic_names=None):
    count = 0
    for ix, topic in enumerate(model.components_):
        if count == no_top_topics:
            break
        if not topic_names or not topic_names[ix]:
            print("\nTopic",(ix + 1))
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(",".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        count += 1

display_topics(svd,vectorizer.get_feature_names(),15,8)

# LDA
# 大写转小写，去掉过长或者过短的文本操作
tokenized_docs = medium['text'].apply(simple_preprocess)
# 生成字典
dictionary = gensim.corpora.Dictionary(tokenized_docs)
# 去掉出现次数低于no_blowed的，去掉出现次数高于no_above
dictionary.filter_extremes(no_below=15,no_above=0.5,keep_n=10000)
# 转换成向量格式
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

lda = models.LdaMulticore(corpus=corpus,num_topics=8,id2word=dictionary,passes=10,workers=4)

lda.print_topics()

columns_names = ['title','url','allTags','readingTime']
# 计算各个类别可能性总和

def compute_dists(top_vec,topic_array):
    dots = np.matmul(topic_array,top_vec)
    input_norm = np.linalg.norm(top_vec)
    co_dists = dots / (input_norm * norms)