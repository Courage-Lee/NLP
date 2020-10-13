import nltk 
nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.corpus import twitter_samples       # 从nltk语料库导入twitter数据集
from nltk.corpus import stopwords    # 导入停用词
from nltk.stem import PorterStemmer   # 导入词干模块
from nltk.tokenize import TweetTokenizer       # 导入分词模块

import pandas as pd
import numpy as np
import re
import string

'''
  数据预处理
'''

# 将数据进行预处理，过滤不需要的数据
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # 去除符号
    tweet = re.sub(r'\$\w*','',tweet)
    tweet = re.sub(r'^RT[\s]+','',tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*','', tweet)
    tweet = re.sub(r'#', '', tweet)

    # 进行分词
    tokenizer=TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tweet_tokens =tokenizer.tokenize(tweet)

    tweet_clean =[]
    for token in tweet_tokens:
        if token not in stopwords_english and token not in string.punctuation:
            stem_token = stemmer.stem(token)
            tweet_clean.append(token)
    
    return tweet_clean

# 计算每个单词分别在positiv 和negative 两种情况下出现的频率
def build_freqs(tweets,label_array):
    """
    Input:
    tweets: tweets数据列表
    label_array : 一个带有情感标签的数组

    output：
    freqs：将每对（单词，情感） 映射到其的字典频率
    """
    # 将label_array 的数组转化为列表 ，和tweets 对应
    label_array_list = np.squeeze(label_array).tolist()

    freqs = {}
    for label, tweet in zip(label_array_list,tweets):
        for word in process_tweet(tweet):
            pair = (word,label)
            if pair in freqs:
                freqs[pair]+=1
            else:
                freqs[pair]=1
    return freqs




'''
函数模型
'''
# 激活函数sigmoid
def sigmoid(x,theta):
    '''
    Input:
        x: a feature vector of dimension (1,n+1)
        theta: your final weight vector
    Output:
        h: the sigmoid of z
    '''

    h = 1/(1+np.exp(-(x.dot(theta))))

    return h


# 梯度下降
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # get 'm', the number of rows in matrix x
    m = np.shape(x)[0]
    
    for i in range(0, num_iters):
        
        
        
        
        # get the sigmoid of z
        h =  sigmoid(x,theta)
        
        # calculate the cost function
        J = (-1) * (y.T.dot(np.log(h)) + ((1-y).T).dot(np.log(1-h))) / m

        # update the weights theta
        theta = theta - (alpha*(x.T.dot(h-y))) / m
        
    ### END CODE HERE ###
    #J = float(J)
    return theta


'''
提取特征:统计一句tweet中，positive 和negative 的数量，形成一个向量
'''

def extract_features(tweet,ferqs):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''

    words = process_tweet(tweet)

    x = np.zeros((1,3))
    x[0,0] = 1

    for word in words:

        if (word,1) in freqs.keys():
            x[0,1] += freqs[(word,1)]

        elif (word,0) in freqs.keys():
            x[0,2] += freqs[(word,0)]

    assert(x.shape==(1,3))

    return x


'''
   模型准确度训练
'''

def predict(tweet,freqs,theta):
    '''
    𝑦_𝑝𝑟𝑒𝑑=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝐱⋅𝜃)
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (n+1,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    x =  extract_features(tweet,freqs)

    y_pred = sigmoid(x,theta)

    return y_pred


def test_logistic_regression(test_tweets,test_labels,freqs,theta):
    """
    Input: 
        test_tweets: a list of tweets
        test_labels: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    y_hat =[]
    

    for tweet in test_tweets:
        y_pred = predict(tweet,freqs,theta)

        if y_pred >0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    j = 0
    for i in range(len(y_hat)):
        if  np.asarray(y_hat)[i] == np.squeeze(test_labels)[i]:
            j +=1

    accuracy  = j/len(y_hat)
    return accuracy

if __name__ == "__main__":
    
    
    '''
    初始化数据
    '''


    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')


    #print(len(all_positive_tweets))

    # 将数据集按照8/2 的比例 分配 训练集和测试集 

    train_positive_tweets = all_positive_tweets[:4000]
    test_positive_tweets =all_positive_tweets[4000:]

    train_negative_tweets = all_negative_tweets[:4000]
    test_negative_tweets =all_negative_tweets[4000:]

    train_tweets = train_positive_tweets + train_negative_tweets
    test_tweets = test_positive_tweets + test_negative_tweets



    # 创建positive标签和negative标签的numpy数组
    train_labels =np.append(np.ones((len(train_positive_tweets),1)),np.zeros((len(train_negative_tweets),1)),axis=0)
    test_labels =np.append(np.ones((len(test_positive_tweets),1)),np.zeros((len(test_negative_tweets),1)),axis=0)

    #创建频率字典
    freqs = build_freqs(train_tweets,train_labels)

    X = np.zeros((len(train_tweets),3))
    for i in range(len(train_tweets)):
        X[i,:]=extract_features(train_tweets[i],freqs)
        
    Y = train_labels

    num_iters  = int(input('请输入迭代次数：'))
    for i in range(num_iters):

        theta = gradientDescent(X,Y,np.zeros((3,1)),1e-9,i)
        #print(theta)
        accuracy = test_logistic_regression(test_tweets,test_labels,freqs,theta)
        if i % 50 == 0:
            print('第%d次的准确率为%f'%(i,accuracy))
    


