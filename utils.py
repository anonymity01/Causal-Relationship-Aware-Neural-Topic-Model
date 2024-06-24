import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
# tf.disable_v2_behavior()
import random
import math
import numpy as np
def data_set(data_url,label_url,label_voc_path):
  """process data input."""
  data = []
  word_count = []
  labels = []
  lines = open(data_url).readlines()
  label_voc={}
  with open(label_voc_path, 'r') as f:
    docs=f.readlines()
  label_num=len(docs)
  for label_ind in range(label_num):
    label_voc[docs[label_ind].strip().split('\t')[0]]=label_ind
  with open(label_url, 'r') as f:
    raw_labels=f.readlines()#[:1000]
  for i in range(len(lines)):
    line=lines[i]
    if not line:
      break
    id_freqs = line.split()
    doc = {}
    count = 0
    for id_freq in id_freqs[1:]:
      items = id_freq.split(':')
      doc[int(items[0])] = int(items[1])
      count += int(items[1])
    if count > 0:
      data.append(doc)
      word_count.append(count)
      class_label=np.zeros(label_num)
      ls=raw_labels[i].strip().split('\t')[1].strip().split(' ')
      if 'russia' in label_url:
        class_label[0]=float(ls[0])/18.0
        ls=ls[1:]
      for l in ls:
        if l.strip() in label_voc:
          class_label[label_voc[l.strip()]]=1
      labels.append(class_label)
  return data, word_count, labels

def create_batches(data_size, batch_size, shuffle=True):
  """create index by batches."""
  batches = []
  ids = list(range(data_size))
  if shuffle:
    random.shuffle(ids)
  for i in range(int(data_size / batch_size)):
    start = i * batch_size
    end = (i + 1) * batch_size
    batches.append(ids[start:end])
  # the batch of which the length is less than batch_size
  rest = data_size % batch_size
  if rest > 0:
    batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
  return batches

def fetch_data(data, label, count, idx_batch, vocab_size):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  data_batch = np.zeros((batch_size, vocab_size))
  label_batch = []
  count_batch = []
  mask = np.zeros(batch_size)
  indices = []
  values = []
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      for word_id, freq in data[doc_id].items():
        data_batch[i, word_id] = freq
      label_batch.append(label[doc_id])
      count_batch.append(count[doc_id])
      mask[i]=1.0
    else:
      label_batch.append(np.zeros(len(label[0])))
      count_batch.append(0)
  return data_batch, label_batch, count_batch, mask

def variable_parser(var_list, prefix):
  """return a subset of the all_variables by prefix."""
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    if varprefix == prefix:
      ret_list.append(var)
    elif prefix in varname:
      ret_list.append(var)
  return ret_list

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def create_counterfactual_contrastive_classifier(inputs, num_classes=2, hidden_dim=100):
    with tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):       
        # Embbeding layer
        hidden = tf.layers.dense(inputs, hidden_dim)
        hidden = tf.layers.batch_normalization(hidden)
        hidden = tf.layers.dropout(hidden, rate=0.25)
        hidden_emb = tf.nn.relu(hidden)
        # Classifier
        outputs = tf.layers.dense(hidden_emb, num_classes)

    return outputs

def linear_LDA(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer =  None
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
    matrix = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.Variable(xavier_init(input_size, output_size))))
    
    output = tf.matmul(inputs, matrix)#no softmax on input, it should already be normalized
    if not no_bias:
      bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
      output = output + bias_term

  return output


def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None,
           weights=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear', reuse=tf.AUTO_REUSE):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer =  tf.truncated_normal_initializer(mean = 0.0, stddev=0.01)
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
    print('[0]',inputs.get_shape()[0].value)
   
    if weights is not None:
      matrix=weights
    else:
      matrix = tf.get_variable('Matrix', [input_size, output_size],initializer=matrix_initializer)
    
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
      output = output + bias_term
    
  return output

def linear_causal(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None,
           weights=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear', reuse=tf.AUTO_REUSE):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer =  tf.truncated_normal_initializer(mean = 0.0, stddev=0.01)
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
    print('[0]',inputs.get_shape()[0].value)

    matrix = tf.get_variable('Matrix')
    matrix_causal = tf.concat([matrix, tf.get_variable('u_Matrix', [input_size-matrix.get_shape()[0], output_size],initializer=matrix_initializer)], 0)
    
    output_causal = tf.matmul(inputs, matrix_causal)
    
    if not no_bias:
      bias_term_causal = tf.get_variable('Bias_causal', [output_size], 
                                initializer=bias_initializer)
      output_causal = output_causal + bias_term_causal
    
  return output_causal

def log_pdf(n_topics, mu, logsig,vec):
    G=[]
    for k in range(n_topics):
        logpdf=-0.5*tf.reshape(tf.reduce_sum(math.log(math.pi*2)+logsig[k:k+1,:]+tf.pow(vec-mu[k:k+1,:],2)/tf.exp(logsig[k:k+1,:]),1),[1,-1])
        G.append(logpdf)
    return tf.concat(G,0)

def mlp(inputs, 
        mlp_hidden=[], 
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):
  """Define an MLP."""
  with tf.variable_scope(scope or 'Linear'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)))
    return res

def mlp_causal(inputs, 
        mlp_hidden=[], 
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):
  """Define an MLP."""
  with tf.variable_scope(scope or 'Linear_causal'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)+'causal'))
    return res 
    
def print_top_words(beta, beta_causal,matrix_A, feature_names,n_topics, n_top_words=50,label_names=None,result_file=None,output_dir='model/'):
    print('---------------Printing the Topics------------------')
    np.savetxt(output_dir+"beta.txt", beta)
    np.savetxt(output_dir+"beta_causal.txt", beta_causal)
    np.savetxt(output_dir+"A.txt", matrix_A)
    f_topic=open(output_dir +"topics.txt", 'w')
    f_label=open(output_dir +"label_topics.txt", 'w')
    for i in range(len(beta)):
      topic_string = " ".join([feature_names[j]
        for j in beta[i].argsort()[:-n_top_words - 1:-1]])
      f_topic.write(topic_string+'\n')
    for i in range(len(beta_causal)):
      topic_string = " ".join([feature_names[j]
        for j in beta_causal[i].argsort()[:-n_top_words - 1:-1]])
      f_label.write(topic_string+'\n')
    f_topic.close()
    f_label.close()
    print('---------------End of Topics------------------')    