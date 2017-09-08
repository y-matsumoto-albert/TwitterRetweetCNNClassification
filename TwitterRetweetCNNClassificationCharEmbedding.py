import tensorflow as tf
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from janome.tokenizer import Tokenizer
import re
from datetime import datetime

class CharBasedTextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 埋め込み層
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 各フィルターサイズに対して畳み込み層とMaxPooling層を作成
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 畳み込み層
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # ReLU活性化関数適用
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # MaxPoolした各アウトプットを結合して1次元配列に変換
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # スコアと予測値
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 交差エントロピー損失関数
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 精度
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class TweetProcessor:
    def __init__(self):
        self.vocab_dict = {}
        self.vocab_dict["<unknown>"] = 0

    # ツイート文のクレンジング
    def filter_sentence(self, sentence):
        sentence = re.sub(r"(?:https?|ftp)://[A-Za-z0-9.-/]*", "", sentence) # urlは削除
        sentence = sentence.replace("RT @", "")
        return sentence
        
    # 文をMAX_CHAR_LENGTHまで0で埋める
    def padding(self, chars, max_length):
        while len(chars) < max_length:
            chars.append(0)
        return np.array(chars)
    
    # 各文字を文字辞書に登録する
    def add_to_vocab_list(self, chars):
        for char in chars:
            if char not in self.vocab_dict:
                self.vocab_dict[char] = len(self.vocab_dict)
    
    # ツイート文を文字IDリストに変換
    def sentence_to_ids(self, sentence):
        ids = []
        for char in list(sentence):
            if char in self.vocab_dict:
                ids.append(self.vocab_dict[char])
            else:
                ids.append(0)
        return ids
        
    def prepare_data(self, file_path_list):
        # ツイートデータの読み込み
        print("ツイートデータ読み込み中..\n\n")
        tweet_dataframes = [pd.read_csv(path, delimiter="\t") for path in file_path_list]
        tweet_dataframe = pd.concat(tweet_dataframes)
        
        # 重複ツイートは削除
        tweet_dataframe.drop_duplicates(['TweetID'])
        
        # リツイートしているツイートは削除
        tweet_dataframe = tweet_dataframe[[tt.startswith("RT @") == False for tt in tweet_dataframe['TweetText']]]
        
        # リツイート閾値より小(ラベル0)とリツイート閾値以上(ラベル1)のデータを1:1比率で抽出
        label1_dataframe = tweet_dataframe[tweet_dataframe['RetweetCount'] >= RETWEET_COUNT_THRESHOLD]
        label0_dataframe = tweet_dataframe[tweet_dataframe['RetweetCount'] < RETWEET_COUNT_THRESHOLD][::int(len(tweet_dataframe[tweet_dataframe['RetweetCount'] < RETWEET_COUNT_THRESHOLD])/len(label1_dataframe))][:len(label1_dataframe)]
        tweet_dataframe = pd.concat([label0_dataframe, label1_dataframe])
        
        # [[ツイート , ラベル([0, 1] or [1, 0])]]のリスト作成
        tweet_label = []
        for tt, rc in zip(tweet_dataframe['TweetText'], tweet_dataframe['RetweetCount']):
            tt = self.filter_sentence(tt)
            # 文字数がMIN_CHAR_LENGTH以上かつMAX_CHAR_LENGTH以下のツイートのみ使用
            if len(list(tt)) < MIN_CHAR_LENGTH or len(list(tt)) > MAX_CHAR_LENGTH:
                continue
            # ツイートの各文字を文字辞書に追加
            self.add_to_vocab_list(list(tt))

            # リツイート数が閾値より小さいものは[1, 0]ラベルを振り、それ以上は[0, 1]ラベルを振る。
            tweet_label.append([tt, [1, 0] if int(rc) < RETWEET_COUNT_THRESHOLD else [0, 1]])
        
        for tl in tweet_label:
            tl[0] = self.sentence_to_ids(tl[0])
            tl[0] = self.padding(tl[0], MAX_CHAR_LENGTH)
            #print(tl[0])
        tweet_label = np.array(tweet_label)
        
        # 学習データ準備
        
        # データをシャッフル
        np.random.shuffle(tweet_label)
        
        #print("tweet_label.shape : ", tweet_label.shape)
        
        # ラベル0のデータとラベル1のデータへ再度分ける
        tweet_label0 = tweet_label[[tl[1][0] == 1 for tl in tweet_label]]
        tweet_label1 = tweet_label[[tl[1][1] == 1 for tl in tweet_label]]
        
        # 8割のラベル0データとラベル1データを学習データに使用
        train_data = np.array([tl for tl in tweet_label0[:int(tweet_label0.shape[0]*0.8), 0]] + [tl for tl in tweet_label1[:int(tweet_label1.shape[0]*0.8), 0]])
        train_label = np.array([tl for tl in tweet_label0[:int(tweet_label0.shape[0]*0.8), 1]] + [tl for tl in tweet_label1[:int(tweet_label1.shape[0]*0.8), 1]])

        # 残り2割のラベル0データとラベル1データを検証データに使用
        test_data = np.array([tl for tl in tweet_label0[int(tweet_label0.shape[0]*0.8):, 0]] + [tl for tl in tweet_label1[int(tweet_label1.shape[0]*0.8):, 0]])
        test_label = np.array([tl for tl in tweet_label0[int(tweet_label0.shape[0]*0.8):, 1]] + [tl for tl in tweet_label1[int(tweet_label1.shape[0]*0.8):, 1]])
        
        return (train_data, train_label, test_data, test_label)
    
RETWEET_COUNT_THRESHOLD = 1
MAX_CHAR_LENGTH = 140 # ツイート文字数最大長。これより長いツイートは学習に使用しない＆分類も不可
MIN_CHAR_LENGTH = 6 # ツイートトークンサイズ最小長。これより短いツイートは学習に使用しない＆分類も不可
word_vec_dim = 50
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
EMBEDDING_SIZE = 128
TWEETDATA_FILEPATH = ["tweet_data/20170604-20170613_JPYUSD.tsv",
                        "tweet_data/20170618-20170627_JPYUSD.tsv",
                        "tweet_data/20170629-20170708_JPYUSD.tsv",
                        "tweet_data/20170706-20170715_JPYUSD.tsv",
                        "tweet_data/20170714-20170722_JPYUSD.tsv",
                        "tweet_data/20170724-20170801_JPYUSD.tsv"]

if __name__ == "__main__":
    tweet_processor = TweetProcessor()
    train_data, train_label, test_data, test_label = tweet_processor.prepare_data(file_path_list = TWEETDATA_FILEPATH)
                            
    sess = tf.InteractiveSession()
    
    with sess.as_default():
        cnn_model = CharBasedTextCNN(num_classes = 2, 
                        vocab_size=len(tweet_processor.vocab_dict), 
                        sequence_length=MAX_CHAR_LENGTH,
                        embedding_size=EMBEDDING_SIZE,                        
                        filter_sizes = FILTER_SIZES,
                        num_filters = NUM_FILTERS)
        print("vocab size : ", len(tweet_processor.vocab_dict))
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn_model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        dropout_keep_prob = 0.5
        
        train_steps = 3000
        # 学習開始
        print("Started Training..")
        local_step = 0
        while True:
            local_step += 1
            
            x_batch = train_data
            y_batch = train_label
            feed_dict = {
              cnn_model.input_x: x_batch,
              cnn_model.input_y: y_batch,
              cnn_model.dropout_keep_prob: dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn_model.loss, cnn_model.accuracy],
                feed_dict)
            now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print("{} : local_step/global_step : {}/{}".format(now, local_step, step))
            if local_step % 1 == 0:
                print("{} : local_step/global_step : {}/{} - accuracy : {}".format(now, local_step, step, accuracy))
                print("{} : local_step/global_step : {}/{} - loss : {}".format(now, local_step, step, loss))
                # Predict for all test data
                predicted = sess.run(cnn_model.predictions, feed_dict={cnn_model.input_x: test_data, cnn_model.dropout_keep_prob: dropout_keep_prob})
                print("{} : local_step/global_step : {}/{} - test data accuracy {}/{} = {}".format(now, local_step, step, len(test_data[predicted == np.argmax(test_label, axis=1)]), len(test_data), len(test_data[predicted == np.argmax(test_label, axis=1)])/len(test_data)))
                # Predict for label 0 data
                predicted0 = sess.run(cnn_model.predictions, feed_dict={cnn_model.input_x: test_data[np.argmax(test_label, axis=1) == 0], cnn_model.dropout_keep_prob: dropout_keep_prob})
                print("{} : local_step/global_step : {}/{} - label 0 accuracy {}/{} = {}".format(now, local_step, step, len(predicted0[predicted0 == 0]), len(test_data[np.argmax(test_label, axis=1) == 0]), len(predicted0[predicted0 == 0])/len(test_data[np.argmax(test_label, axis=1) == 0])))
                # predict for label 1 data
                predicted1 = sess.run(cnn_model.predictions, feed_dict={cnn_model.input_x: test_data[np.argmax(test_label, axis=1) == 1], cnn_model.dropout_keep_prob: dropout_keep_prob})
                print("{} : local_step/global_step : {}/{} - label 1 accuracy {}/{} = {}".format(now, local_step, step, len(predicted1[predicted1 == 1]), len(test_data[np.argmax(test_label, axis=1) == 1]), len(predicted1[predicted1 == 1])/len(test_data[np.argmax(test_label, axis=1) == 1])))
          
            if (step + 1) % 1000 == 0 or (step + 1) == train_steps:
                # save model
                saver.save(sess, "TwitterRetweetClassification_char_ver_{}".format(step+1), global_step=step)
          
            if step >= train_steps:
                break