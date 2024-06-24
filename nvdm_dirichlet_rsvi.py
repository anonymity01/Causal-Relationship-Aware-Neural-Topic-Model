"""NVDM Tensorflow implementation by Yishu Miao, adapted to work with the Dirichlet distribution by Sophie Burkhardt"""
from __future__ import print_function
import warnings
import os
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
# tf.disable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import math
import utils as utils
import sys
import argparse
import pickle

from mask import *

np.random.seed(0)
tf.set_random_seed(0)

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('n_hidden', 100, 'Size of each hidden layer.')
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity of the MLP.')
flags.DEFINE_string('summaries_dir','summaries','where to save the summaries')
FLAGS = flags.FLAGS

class NVDM(object):
    """ Neural Variational Document Model -- BOW VAE.
    """
    def __init__(self, 
                 analytical,
                 vocab_size,
                 n_hidden,
                 n_topic, 
                 n_sample,
                 learning_rate, 
                 batch_size,
                 non_linearity,
                 adam_beta1,
                 adam_beta2,
                 B,
                 dir_prior,
                 correction,
                 y_dim,
                 z2_dim):
        tf.reset_default_graph()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.dag_dim = n_topic+y_dim
        self.y_dim = y_dim
        self.z2_dim = z2_dim

        lda=False
        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.warm_up = tf.placeholder(tf.float32, (), name='warm_up')  # warm up
        self.B=tf.placeholder(tf.int32, (), name='B')
        self.adam_beta1=adam_beta1
        self.adam_beta2=adam_beta2
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.min_alpha = tf.placeholder(tf.float32,(), name='min_alpha')
        self.label = tf.placeholder(tf.float32, [None, y_dim], name='input_label')
        self.dataset_flag=tf.placeholder(tf.int32, (), name='dataset_flag')

        # self.dag = DagLayer(in_features=self.dag_dim, out_features=self.dag_dim, use_bias=True)
        # self.attn = Attention(self.dag_dim)
        # self.mask_z = MaskLayer(self.dag_dim,z2_dim=self.z2_dim)
        # self.mask_u = MaskLayer(self.n_topics+self.y_dim,self.flag,z2_dim=1)


        # encoder
        with tf.variable_scope('encoder'): 
          self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity)
          self.enc_vec = tf.nn.dropout(self.enc_vec,self.keep_prob)
          self.mean = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean'))
          self.alpha = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(self.mean)))
          # print(tf.shape(self.alpha))
          # exit()
         
          self.enc_vec_causal = utils.mlp_causal(tf.concat([self.alpha, self.label], axis=1), [self.n_hidden], self.non_linearity)
          self.enc_vec_causal = tf.nn.dropout(self.enc_vec_causal,self.keep_prob)
          self.mean_causal = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec_causal, self.dag_dim, scope='mean_causal'))
          # self.q_m = self.mean_causal
          # self.q_v = tf.ones([self.batch_size, self.dag_dim, self.z2_dim])

          # self.cp_m = tf.concat([self.alpha, self.label], axis=1)
          # self.cp_m = tf.expand_dims(self.cp_m, axis=2)
          # self.cp_m = tf.tile(self.cp_m, [1, self.dag_dim, self.z2_dim])
          # self.cp_v = tf.ones([self.batch_size, self.dag_dim, self.z2_dim])

          # self.q_m = tf.reshape(self.q_m, [self.batch_size, self.dag_dim, self.z2_dim])
          # # self.q_v = tf.reshape(self.q_v, [self.batch_size, self.dag_dim, self.z2_dim])

          # self.decode_m, self.decode_v = self.dag.calculate_dag(self.q_m, tf.ones([self.batch_size, self.dag_dim, self.z2_dim]))

          # self.decode_m = tf.reshape(self.decode_m, [self.batch_size, self.dag_dim, self.z2_dim])
          # # decode_v = decode_v

          # self.m_zm, self.m_zv = self.dag.mask_z(self.decode_m), self.decode_v

          # self.f_z = tf.reshape(self.mask_z.mix(self.m_zm), [self.batch_size, self.dag_dim, self.z2_dim])

          # self.e_tilde = self.attn.attention(tf.reshape(self.decode_m, [self.batch_size, self.dag_dim, self.z2_dim]), tf.reshape(self.q_m, [self.batch_size, self.dag_dim, self.z2_dim]))[0]

          # self.f_z1 = self.f_z + self.e_tilde
          # self.lambdav=0.001
          # self.z_given_dag = conditional_sample_gaussian(self.f_z1, self.m_zv * self.lambdav)
          # self.z_given_dag = tf.reshape(self.z_given_dag, [self.batch_size, self.dag_dim*self.z2_dim])
          
          # self.z_given_dag = utils.linear(self.z_given_dag, self.dag_dim, scope='z_given_dag_causal')
        

          self.alpha_causal = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(self.mean_causal)))

          #Dirichlet prior alpha0
          self.prior = tf.ones((batch_size,self.n_topic), dtype=tf.float32, name='prior')*dir_prior
          self.prior_causal = tf.ones((batch_size,self.dag_dim), dtype=tf.float32, name='prior')*dir_prior

          
          
          
          self.analytical_kld = tf.lgamma(tf.reduce_sum(self.alpha,axis=1))-tf.lgamma(tf.reduce_sum(self.prior,axis=1))
          self.analytical_kld-=tf.reduce_sum(tf.lgamma(self.alpha),axis=1)
          self.analytical_kld+=tf.reduce_sum(tf.lgamma(self.prior),axis=1)
          minus = self.alpha-self.prior
          test = tf.reduce_sum(tf.multiply(minus,tf.digamma(self.alpha)-tf.reshape(tf.digamma(tf.reduce_sum(self.alpha,1)),(batch_size,1))),1)
          self.analytical_kld+=test
          self.analytical_kld = self.mask*self.analytical_kld  # mask paddings

          # ---------causal-----------
          self.analytical_kld_causal = tf.lgamma(tf.reduce_sum(self.alpha_causal,axis=1))-tf.lgamma(tf.reduce_sum(self.prior_causal,axis=1))
          self.analytical_kld_causal-=tf.reduce_sum(tf.lgamma(self.alpha_causal),axis=1)
          self.analytical_kld_causal+=tf.reduce_sum(tf.lgamma(self.prior_causal),axis=1)
          minus_causal = self.alpha_causal-self.prior_causal
          test_causal = tf.reduce_sum(tf.multiply(minus_causal,tf.digamma(self.alpha_causal)-tf.reshape(tf.digamma(tf.reduce_sum(self.alpha_causal,1)),(batch_size,1))),1)
          self.analytical_kld_causal+=test_causal
          self.analytical_kld_causal = self.mask*self.analytical_kld_causal  # mask paddings

        with tf.variable_scope('decoder'):
          if self.n_sample ==1:  # single sample
            #sample gammas
            gam = tf.squeeze(tf.random_gamma(shape = (1,),alpha=self.alpha+tf.to_float(self.B)))
            #reverse engineer the random variables used in the gamma rejection sampler
            eps = tf.stop_gradient(calc_epsilon(gam,self.alpha+tf.to_float(self.B)))
            #uniform variables for shape augmentation of gamma
            u = tf.random_uniform((self.B,batch_size,self.n_topic))
            with tf.variable_scope('prob'):
                #this is the sampled gamma for this document, boosted to reduce the variance of the gradient
                self.doc_vec = gamma_h_boosted(eps,u,self.alpha,self.B)
                # print('doc_vec')
               
                #normalize
                self.doc_vec = tf.div(gam,tf.reshape(tf.reduce_sum(gam,1), (-1, 1)))
                # print(self.doc_vec)
                # exit()
                self.doc_vec.set_shape(self.alpha.get_shape()) 

            #reconstruction
            if lda:
              logits = tf.log(tf.clip_by_value(utils.linear_LDA(self.doc_vec, self.vocab_size, scope='projection',no_bias=True),1e-10,1.0))
            else:
              logits = tf.nn.log_softmax(tf.contrib.layers.batch_norm(utils.linear(self.doc_vec, self.vocab_size, scope='projection',no_bias=True)))
            self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)
            
            dir1=tf.contrib.distributions.Dirichlet(self.prior)
            dir2=tf.contrib.distributions.Dirichlet(self.alpha)
            self.kld = tf.abs(dir2.log_prob(self.doc_vec)-dir1.log_prob(self.doc_vec))
            max_kld_sampled = tf.argmax(self.kld,0)
            # print(self.kld)
            # print(self.recons_loss)
            # exit()

            # ---------causal decoder---------
            self.dag = DagLayer(in_features=self.dag_dim, out_features=self.dag_dim, use_bias=True)
            self.attn = Attention(self.dag_dim)
            self.mask_z = MaskLayer(self.dag_dim,z2_dim=self.z2_dim)
            #sample gammas
            gam_causal = tf.squeeze(tf.random_gamma(shape = (1,),alpha=self.alpha_causal+tf.to_float(self.B)))
            #reverse engineer the random variables used in the gamma rejection sampler
            eps_causal = tf.stop_gradient(calc_epsilon(gam_causal,self.alpha_causal+tf.to_float(self.B)))
            #uniform variables for shape augmentation of gamma
            u_causal = tf.random_uniform((self.B,batch_size,self.dag_dim))
            with tf.variable_scope('prob_causal'):
                #this is the sampled gamma for this document, boosted to reduce the variance of the gradient
                # gam_causal = gamma_h_boosted(eps_causal,u_causal,self.alpha_causal,self.B)
                self.doc_vec_causal = gamma_h_boosted(eps_causal,u_causal,self.alpha_causal,self.B)
                # print(self.doc_vec_causal.get_shape())
                # exit()

                self.q_m = tf.contrib.layers.batch_norm(utils.linear(tf.reshape(self.doc_vec_causal, [self.batch_size, self.dag_dim]), self.dag_dim*self.z2_dim, scope='q_m'))
                self.q_v = tf.ones([self.batch_size, self.dag_dim, self.z2_dim])

                self.cp_m = tf.concat([tf.div(self.doc_vec_causal,tf.reshape(tf.reduce_sum(self.doc_vec_causal,1), (-1, 1))), self.label], axis=1)
                self.cp_m = tf.expand_dims(self.cp_m, axis=2)
                self.cp_m = tf.tile(self.cp_m, [1, self.dag_dim, self.z2_dim])
                self.cp_v = tf.ones([self.batch_size, self.dag_dim, self.z2_dim])

                self.q_m = tf.reshape(self.q_m, [self.batch_size, self.dag_dim, self.z2_dim])
                self.q_v = tf.reshape(self.q_v, [self.batch_size, self.dag_dim, self.z2_dim])

                # self.decode_m, self.decode_v = utils.calculate_dag(self.q_m, tf.ones([self.batch_size, self.dag_dim, self.z2_dim]),self.dag_dim)
                self.decode_m, self.decode_v = self.dag.calculate_dag(self.q_m, tf.ones([self.batch_size, self.dag_dim, self.z2_dim]))

                self.decode_m = tf.reshape(self.decode_m, [self.batch_size, self.dag_dim, self.z2_dim])
                # decode_v = decode_v

                # self.m_zm,self.dag_param = utils.mask_z(self.decode_m,self.dag_dim)
                self.m_zv = self.decode_v
                self.m_zm, self.m_zv = self.dag.mask_z(self.decode_m), self.decode_v

                self.f_z = tf.reshape(self.mask_z.mix(self.m_zm), [self.batch_size, self.dag_dim, self.z2_dim])

                self.e_tilde = self.attn.attention(tf.reshape(self.decode_m, [self.batch_size, self.dag_dim, self.z2_dim]), tf.reshape(self.q_m, [self.batch_size, self.dag_dim, self.z2_dim]))[0]

                self.f_z1 = self.f_z + self.e_tilde
                self.lambdav=0.001
                self.z_given_dag = conditional_sample_gaussian(self.f_z1, self.m_zv * self.lambdav)
                self.z_given_dag = tf.reshape(self.z_given_dag, [self.batch_size, self.dag_dim*self.z2_dim])
                self.z_given_dag = tf.contrib.layers.batch_norm(utils.linear(self.z_given_dag, self.dag_dim, scope='z_given_dag_causal'))

                # self.z_given_dag = utils.linear(self.m_zm, self.dag_dim, scope='z_given_dag_causal')
                # print('doc_vec')
                self.doc_vec_causal = tf.nn.softmax(self.z_given_dag)
                #normalize
                # self.doc_vec_causal = tf.div(gam_causal,tf.reshape(tf.reduce_sum(gam_causal,1), (-1, 1)))
                # print(self.doc_vec)
                # exit()
                self.doc_vec_causal.set_shape(self.alpha_causal.get_shape()) 

            #reconstruction
            if lda:
              logits = tf.log(tf.clip_by_value(utils.linear_LDA(self.doc_vec_causal, self.vocab_size, scope='projection',no_bias=True),1e-10,1.0))
            else:
              logits_causal = tf.nn.log_softmax(tf.contrib.layers.batch_norm(utils.linear_causal(self.doc_vec_causal, self.vocab_size, scope='projection',no_bias=True)))
            self.recons_loss_causal = -tf.reduce_sum(tf.multiply(logits_causal, self.x), 1)
            
            dir1_causal=tf.contrib.distributions.Dirichlet(self.prior_causal)
            dir2_causal=tf.contrib.distributions.Dirichlet(self.alpha_causal)
            self.kld_causal = tf.abs(dir2_causal.log_prob(self.doc_vec_causal)-dir1_causal.log_prob(self.doc_vec_causal))
            max_kld_sampled_causal = tf.argmax(self.kld_causal,0)
            self.mask_kl = tf.zeros([1])
            for i in range(self.dag_dim):
              self.mask_kl = self.mask_kl + kl_normal(self.f_z1[:, i, :], self.cp_v[:, i, :], self.cp_m[:, i, :], self.cp_v[:, i, :])
              self.kld_causal = self.kld_causal + kl_normal(self.decode_m[:, i, :], self.cp_v[:, i, :], self.cp_m[:, i, :], self.cp_v[:, i, :])
                # self.kld_causal = self.kld_causal + tf.reduce_sum(self.alpha)

            # for weight in self.dag.weights:
            #   if weight.name == 'A:0':
            # with tf.variable_scope("encoder/mask_z", reuse=True):
            # dec_vars = utils.variable_parser(tf.trainable_variables(), 'decoder')
            # print(dec_vars)
            # exit()
            self.dag_param = self.dag.A
            # print(self.dag_param)
            # exit()
            # self.dag_param=tf.zeros([237,237])
            self.h_a = _h_A(self.dag_param, self.dag_dim)

            # -----do causal------
            causal_candidate_idx = tf.where(self.dag_param[self.n_topic:][:self.n_topic] >= 0)
            anti_causal_candidate_idx = tf.where(self.dag_param[:self.n_topic][self.n_topic:] >= 0)
            n_causal_candidate = causal_candidate_idx.get_shape()[0]
            n_anti_causal_candidate = anti_causal_candidate_idx.get_shape()[0]
            self.cross_entropy_loss = tf.zeros((1))

            if n_causal_candidate > 0:
              sampled_causal = np.random.randint(0, n_causal_candidate)
              self.cross_entropy_loss += clr(self.label, self.data, causal_candidate_idx[sampled_causal][0], tf.ones((tf.shape(label)[0]), dtype=tf.int64))

            if n_anti_causal_candidate > 0:
              sampled_causal = np.random.randint(0, n_anti_causal_candidate)
              self.cross_entropy_loss += clr(self.label, self.data, anti_causal_candidate_idx[sampled_causal][1], tf.zeros((tf.shape(label)[0]), dtype=tf.int64))

        self.objective = self.recons_loss + self.warm_up*self.kld + self.recons_loss_causal + self.warm_up*self.kld_causal + self.warm_up*self.mask_kl + 3 * self.h_a + 0.5 * self.h_a * self.h_a + self.cross_entropy_loss
        #self.objective = self.recons_loss + self.warm_up*self.analytical_kld
        # self.true_objective = self.recons_loss_causal + self.kld_causal
        self.true_objective = self.recons_loss + self.kld + self.recons_loss_causal + self.kld_causal + self.mask_kl
       
        # self.analytical_objective = self.recons_loss_causal+self.analytical_kld_causal
        self.analytical_objective = self.recons_loss+self.analytical_kld + self.recons_loss_causal+self.analytical_kld_causal


        # self.objective = self.recons_loss + self.warm_up*self.kld
        # #self.objective = self.recons_loss + self.warm_up*self.analytical_kld
        # self.true_objective = self.recons_loss + self.kld
       
        # self.analytical_objective = self.recons_loss+self.analytical_kld
       
        fullvars = tf.trainable_variables()

        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')
        # print(enc_vars)
        # print('------------')
        # print(dec_vars)
        # print('------------')
        # print(fullvars)
        # exit()`
        # dec_vars.append(self.dag_param)
        # print(dec_vars)
        # exit()
       
        #this is the standard gradient for the reconstruction network
        dec_grads = tf.gradients(self.objective, dec_vars)
        # dag_grads = tf.gradients(self.objective, self.dag_param)

        # full_grads = tf.gradients(self.objective, fullvars)
        
        #####################################################
        #Now calculate the gradient for the encoding network#
        #####################################################
       
        #redefine kld and recons_loss for proper gradient back propagation
        if self.n_sample ==1:
          gammas = gamma_h_boosted(eps,u,self.alpha,self.B)
          self.doc_vec = tf.div(gammas,tf.reshape(tf.reduce_sum(gammas,1), (-1, 1)))
          self.doc_vec.set_shape(self.alpha.get_shape())

          gammas_causal = gamma_h_boosted(eps_causal,u_causal,self.alpha_causal,self.B)
          # self.q_m = gammas_causal
          self.q_m = tf.contrib.layers.batch_norm(utils.linear(tf.reshape(gammas_causal, [self.batch_size, self.dag_dim]), self.dag_dim*self.z2_dim, scope='q_m'))
          self.q_v = tf.ones([self.batch_size, self.dag_dim, self.z2_dim])

          # self.cp_m = tf.concat([tf.div(gammas_causal,tf.reshape(tf.reduce_sum(gammas_causal,1), (-1, 1))), self.label], axis=1)
          # self.cp_m = tf.expand_dims(self.cp_m, axis=2)
          # self.cp_m = tf.tile(self.cp_m, [1, self.dag_dim, self.z2_dim])
          # self.cp_v = tf.ones([self.batch_size, self.dag_dim, self.z2_dim])

          self.q_m = tf.reshape(self.q_m, [self.batch_size, self.dag_dim, self.z2_dim])
          self.q_v = tf.reshape(self.q_v, [self.batch_size, self.dag_dim, self.z2_dim])

          # self.decode_m, self.decode_v = utils.calculate_dag(self.q_m, tf.ones([self.batch_size, self.dag_dim, self.z2_dim]),self.dag_dim)
          self.decode_m, self.decode_v = self.dag.calculate_dag(self.q_m, tf.ones([self.batch_size, self.dag_dim, self.z2_dim]))

          self.decode_m = tf.reshape(self.decode_m, [self.batch_size, self.dag_dim, self.z2_dim])
          # decode_v = decode_v

          # self.m_zm,self.dag_param = utils.mask_z(self.decode_m,self.dag_dim)
          self.m_zv = self.decode_v
          self.m_zm, self.m_zv = self.dag.mask_z(self.decode_m), self.decode_v

          self.f_z = tf.reshape(self.mask_z.mix(self.m_zm), [self.batch_size, self.dag_dim, self.z2_dim])

          self.e_tilde = self.attn.attention(tf.reshape(self.decode_m, [self.batch_size, self.dag_dim, self.z2_dim]), tf.reshape(self.q_m, [self.batch_size, self.dag_dim, self.z2_dim]))[0]

          self.f_z1 = self.f_z + self.e_tilde
          self.lambdav=0.001
          self.z_given_dag = conditional_sample_gaussian(self.f_z1, self.m_zv * self.lambdav)
          self.z_given_dag = tf.reshape(self.z_given_dag, [self.batch_size, self.dag_dim*self.z2_dim])
                
          self.z_given_dag = tf.contrib.layers.batch_norm(utils.linear(self.z_given_dag, self.dag_dim, scope='z_given_dag_causal'))

          # self.z_given_dag = utils.linear(self.m_zm, self.dag_dim, scope='z_given_dag_causal')
          # print('doc_vec')
          self.doc_vec_causal = tf.nn.softmax(self.z_given_dag)
          # self.doc_vec_causal = tf.div(gammas_causal,tf.reshape(tf.reduce_sum(gammas_causal,1), (-1, 1)))
          self.doc_vec_causal.set_shape(self.alpha_causal.get_shape())

          with tf.variable_scope("decoder", reuse=True):
              logits2 = tf.nn.log_softmax(tf.contrib.layers.batch_norm(utils.linear(self.doc_vec, self.vocab_size, scope='projection',no_bias=True)))
              self.recons_loss2 = -tf.reduce_sum(tf.multiply(logits2, self.x), 1)
              prior_sample = tf.squeeze(tf.random_gamma(shape = (1,),alpha=self.prior))
              prior_sample = tf.div(prior_sample,tf.reshape(tf.reduce_sum(prior_sample,1), (-1, 1)))
             
              self.kld2 = tf.contrib.distributions.Dirichlet(self.alpha).log_prob(self.doc_vec)-tf.contrib.distributions.Dirichlet(self.prior).log_prob(self.doc_vec)
              # ---------causal---------
              logits2_causal = tf.nn.log_softmax(tf.contrib.layers.batch_norm(utils.linear_causal(self.doc_vec_causal, self.vocab_size, scope='projection',no_bias=True)))
              self.recons_loss2_causal = -tf.reduce_sum(tf.multiply(logits2_causal, self.x), 1)
              prior_sample_causal = tf.squeeze(tf.random_gamma(shape = (1,),alpha=self.prior_causal))
              prior_sample_causal = tf.div(prior_sample_causal,tf.reshape(tf.reduce_sum(prior_sample_causal,1), (-1, 1)))
             
              self.kld2_causal = tf.contrib.distributions.Dirichlet(self.alpha_causal).log_prob(self.doc_vec_causal)-tf.contrib.distributions.Dirichlet(self.prior_causal).log_prob(self.doc_vec_causal)
        # else:
        #   with tf.variable_scope("decoder", reuse=True):
        #     recons_loss_list2 = []
        #     kld_list2 = []
            
        #     for i in range(self.n_sample):
        #       curr_gam = gam[i]
        #       eps = tf.stop_gradient(calc_epsilon(curr_gam,self.alpha+tf.to_float(self.B)))
        #       curr_u = u[i]
        #       self.doc_vec = gamma_h_boosted(eps,curr_u,self.alpha,self.B)
        #       self.doc_vec = tf.div(self.doc_vec,tf.reshape(tf.reduce_sum(self.doc_vec,1), (-1, 1)))
        #       self.doc_vec.set_shape(self.alpha.get_shape())
        #       if lda:
        #         logits2 = tf.log(tf.clip_by_value(utils.linear_LDA(self.doc_vec, self.vocab_size, scope='projection',no_bias=True),1e-10,1.0))
        #       else:
        #         logits2 = tf.nn.log_softmax(tf.contrib.layers.batch_norm(utils.linear(self.doc_vec, self.vocab_size, scope='projection',no_bias=True),scope ='projection'))
        #       loss = -tf.reduce_sum(tf.multiply(logits2, self.x), 1)
        #       recons_loss_list2.append(loss)
        #       prior_sample = tf.squeeze(tf.random_gamma(shape = (1,),alpha=self.prior))
        #       prior_sample = tf.div(prior_sample,tf.reshape(tf.reduce_sum(prior_sample,1), (-1, 1)))
        #       kld2 = tf.contrib.distributions.Dirichlet(self.alpha).log_prob(self.doc_vec)-tf.contrib.distributions.Dirichlet(self.prior).log_prob(self.doc_vec)
        #       kld_list2.append(kld2)
        #     self.recons_loss2 = tf.add_n(recons_loss_list2) / self.n_sample
            
        #     self.kld2 = tf.add_n(kld_list2)/self.n_sample
            
        # if analytical:
        #   kl_grad = tf.gradients(self.analytical_kld,enc_vars)
        # else:
        kl_grad = tf.gradients(self.kld2,enc_vars)
        kl_grad_causal = tf.gradients(self.kld2_causal,enc_vars)
            
        #this is the gradient we would use if the rejection sampler for the Gamma would always accept
        
        g_rep = tf.gradients(self.recons_loss2,enc_vars)
        g_rep_causal = tf.gradients(self.recons_loss2_causal,enc_vars)
        
        #now define the gradient for the correction part
        
        # for var in enc_vars:
        #   print(var)
        # exit()
        logpi_gradient = [tf.squeeze(separate_gradients(log_q(gamma_h(eps, self.alpha+tf.to_float(self.B),1.), self.alpha+tf.to_float(self.B), 1.)+tf.log(dh(eps, self.alpha+tf.to_float(self.B), 1.)),var)) for var in enc_vars[:5]]

        logpi_gradient_causal = [tf.squeeze(separate_gradients(log_q(gamma_h(eps_causal, self.alpha_causal+tf.to_float(self.B),1.), self.alpha_causal+tf.to_float(self.B), 1.)+tf.log(dh(eps_causal, self.alpha_causal+tf.to_float(self.B), 1.)),var)) for var in enc_vars[5:]]
      
        # print(logpi_gradient)
        # exit()
        #now multiply with the reconstruction loss
        reshaped1 = tf.reshape(self.recons_loss,(batch_size,1))
        reshaped2 = tf.reshape(self.recons_loss,(batch_size,1,1))
        reshaped21 = tf.reshape(self.kld,(batch_size,1))
        reshaped22 = tf.reshape(self.kld,(batch_size,1,1))
        
        g_cor = []
        g_cor2 = []
        g_cor2.append(tf.multiply(reshaped22,logpi_gradient[0]))
        g_cor2.append(tf.multiply(reshaped21,logpi_gradient[1]))
        g_cor2.append(tf.multiply(reshaped22,logpi_gradient[2]))
        g_cor2.append(tf.multiply(reshaped21,logpi_gradient[3]))
        g_cor.append(tf.multiply(reshaped2,logpi_gradient[0]))
        g_cor.append(tf.multiply(reshaped1,logpi_gradient[1]))
        g_cor.append(tf.multiply(reshaped2,logpi_gradient[2]))
        g_cor.append(tf.multiply(reshaped1,logpi_gradient[3]))
        #sum over instances
        g_cor = [tf.reduce_sum(gc,0) for gc in g_cor]
        g_cor2 = [tf.reduce_sum(gc,0) for gc in g_cor2]
        
        # ------causal------
        reshaped1_causal = tf.reshape(self.recons_loss_causal,(batch_size,1))
        reshaped2_causal = tf.reshape(self.recons_loss_causal,(batch_size,1,1))
        reshaped21_causal = tf.reshape(self.kld_causal,(batch_size,1))
        reshaped22_causal = tf.reshape(self.kld_causal,(batch_size,1,1))
        
        g_cor_causal = []
        g_cor2_causal = []
        g_cor2_causal.append(tf.multiply(reshaped22_causal,logpi_gradient_causal[0]))
        g_cor2_causal.append(tf.multiply(reshaped21_causal,logpi_gradient_causal[1]))
        g_cor2_causal.append(tf.multiply(reshaped22_causal,logpi_gradient_causal[2]))
        g_cor2_causal.append(tf.multiply(reshaped21_causal,logpi_gradient_causal[3]))
        g_cor_causal.append(tf.multiply(reshaped2_causal,logpi_gradient_causal[0]))
        g_cor_causal.append(tf.multiply(reshaped1_causal,logpi_gradient_causal[1]))
        g_cor_causal.append(tf.multiply(reshaped2_causal,logpi_gradient_causal[2]))
        g_cor_causal.append(tf.multiply(reshaped1_causal,logpi_gradient_causal[3]))
        #sum over instances
        g_cor_causal = [tf.reduce_sum(gc,0) for gc in g_cor_causal]
        g_cor2_causal = [tf.reduce_sum(gc,0) for gc in g_cor2_causal]
      
        #finally sum up the three parts
        if not correction:
          enc_grads = [g_r+self.warm_up*g_e+g_r_causal+self.warm_up*g_e_causal for g_r,g_c,g_e,g_r_causal,g_c_causal,g_e_causal in zip(g_rep,g_cor,kl_grad,g_rep_causal,g_cor_causal,kl_grad_causal)]
          # enc_grads = [g_r+self.warm_up*g_e for g_r,g_c,g_e in zip(g_rep,g_cor,kl_grad)]
        else:
          enc_grads = [g_r+g_c+g_c2+self.warm_up*g_e+g_r_causal+g_c_causal+g_c2_causal+self.warm_up*g_e_causal for g_r,g_c,g_c2,g_e,g_r_causal,g_c_causal,g_c2_causal,g_e_causal in zip(g_rep,g_cor,g_cor2,kl_grad,g_rep_causal,g_cor_causal,g_cor2_causal,kl_grad_causal)]
          # enc_grads = [g_r+g_c+g_c2+self.warm_up*g_e for g_r,g_c,g_c2,g_e in zip(g_rep,g_cor,g_cor2,kl_grad)]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.adam_beta1,beta2=self.adam_beta2)
        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))
        self.optim_all = optimizer.apply_gradients(list(zip(enc_grads, enc_vars))+list(zip(dec_grads, dec_vars)))

    def get_dir_z(data, do_label):
        # encoder
        with tf.variable_scope('encoder'): 
          enc_vec = utils.mlp(data, [self.n_hidden], self.non_linearity)
          enc_vec = tf.nn.dropout(enc_vec,self.keep_prob)
          mean = tf.contrib.layers.batch_norm(utils.linear(enc_vec, self.n_topic, scope='mean'))
          alpha = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(mean)))
          # print(tf.shape(self.alpha))
          # exit()
         
          enc_vec_causal = utils.mlp_causal(tf.concat([alpha, do_label], axis=1), [self.n_hidden], self.non_linearity)
          enc_vec_causal = tf.nn.dropout(enc_vec_causal,self.keep_prob)
          mean_causal = tf.contrib.layers.batch_norm(utils.linear(enc_vec_causal, self.dag_dim, scope='mean_causal'))

          alpha_causal = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(mean_causal)))

        

        with tf.variable_scope('decoder'):
          if self.n_sample ==1:  # single sample
            #sample gammas
            gam_causal = tf.squeeze(tf.random_gamma(shape = (1,),alpha=alpha_causal+tf.to_float(self.B)))
            #reverse engineer the random variables used in the gamma rejection sampler
            eps_causal = tf.stop_gradient(calc_epsilon(gam_causal,alpha_causal+tf.to_float(self.B)))
            #uniform variables for shape augmentation of gamma
            u_causal = tf.random_uniform((self.B,batch_size,self.dag_dim))
            with tf.variable_scope('prob_causal'):
                #this is the sampled gamma for this document, boosted to reduce the variance of the gradient
                doc_vec_causal = gamma_h_boosted(eps_causal,u_causal,self.alpha_causal,self.B)

                q_m = tf.contrib.layers.batch_norm(utils.linear(tf.reshape(self.doc_vec_causal, [self.batch_size, self.dag_dim]), self.dag_dim*self.z2_dim, scope='q_m'))
                q_v = tf.ones([self.batch_size, self.dag_dim, self.z2_dim])

                q_m = tf.reshape(self.q_m, [self.batch_size, self.dag_dim, self.z2_dim])
                q_v = tf.reshape(self.q_v, [self.batch_size, self.dag_dim, self.z2_dim])

                decode_m, decode_v = self.dag.calculate_dag(self.q_m, tf.ones([self.batch_size, self.dag_dim, self.z2_dim]))

                decode_m = tf.reshape(decode_m, [self.batch_size, self.dag_dim, self.z2_dim])
                m_zm, m_zv = self.dag.mask_z(self.decode_m), decode_v

                f_z = tf.reshape(self.mask_z.mix(self.m_zm), [self.batch_size, self.dag_dim, self.z2_dim])

                e_tilde = self.attn.attention(tf.reshape(decode_m, [self.batch_size, self.dag_dim, self.z2_dim]), tf.reshape(q_m, [self.batch_size, self.dag_dim, self.z2_dim]))[0]

                f_z1 = f_z + e_tilde
                z_given_dag = conditional_sample_gaussian(f_z1, m_zv * self.lambdav)
                z_given_dag = tf.reshape(z_given_dag, [self.batch_size, self.dag_dim*self.z2_dim])
                z_given_dag = tf.contrib.layers.batch_norm(utils.linear(z_given_dag, self.dag_dim, scope='z_given_dag_causal'))
                doc_vec_causal = tf.nn.softmax(z_given_dag)
                #normalize
                doc_vec_causal.set_shape(alpha_causal.get_shape()) 
        return doc_vec_causal

    def clr(label, data, sampled_u, expected_relation):
        do_causal_matrix = tf.eye(tf.shape(label)[1])
        do_label = tf.abs(label - do_causal_matrix[sampled_u])
        do_ouput_z = get_dir_z(data, do_label)[:, :self.n_topic]
        classify_ouputs = utils.create_counterfactual_contrastive_classifier(tf.reshape(do_ouput_z, [tf.shape(label)[0], -1]))
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classify_ouputs, labels=expected_relation))
        # lvae.criterion(classify_ouputs, expected_relation)
        return cross_entropy_loss
        
def matrix_power(matrix, power):
    result = tf.identity(matrix)
    for _ in range(power - 1):
        result = tf.linalg.matmul(result, matrix)
    return result

def matrix_poly(matrix, d):
    x = tf.eye(d) + tf.divide(matrix, d)
    return matrix_power(x, d)

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = tf.linalg.trace(expm_A) - m
    return h_A

def kl_normal(qm, qv, pm, pv):
  kl = 0.5 * tf.reduce_sum(tf.square(qm - pm), axis=1)
  return kl
def log_dirichlet(x,alpha):
  first=-tf.reduce_sum(tf.lgamma(alpha),1)+tf.lgamma(tf.reduce_sum(alpha,1))
  second = tf.reduce_sum((alpha-1.)*tf.log(x),1)
  return first+second
  
"""
calculates the jacobian between a vector and some other tensor
"""
def jacobian(y_flat, x):
    n = tf.shape(y_flat)[0]
    
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x))),
        loop_vars)
    return jacobian.stack()
    
"""
calculates the jacobian between a 2-dimensional matrix and some other tensor
"""
def jacobian2(y_flat, x):
    n = tf.shape(y_flat)[0]
    m=tf.shape(y_flat)[1]
    
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    
    def body(j, result):
        loop_vars_inner_loop = [
            loop_vars[0],
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=m),
        ]
        _,_,row = tf.while_loop(lambda i,k, _: (k<m),
                    lambda i,k, row:(i,k+1,row.write(k,tf.gradients(y_flat[i][k], x))),
                    loop_vars_inner_loop)
        result = result.write(j, row.stack())
        return (j+1,result)
    
    _, jacobian = tf.while_loop(
        lambda j, _: (j<n),
        body,
        loop_vars)
    return jacobian.stack()
    
"""
returns the gradient for each data instance separately
"""
def separate_gradients(y_flat, x):
    n = tf.shape(y_flat)[0]
    
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j],x))),
        loop_vars)
    return jacobian.stack()

# Log density of Ga(alpha, beta)
def log_q(z, alpha, beta):
    return -tf.lgamma(alpha) + alpha * tf.log(beta) \
           + (alpha - 1) * tf.log(z) - beta * z

# Log density of N(0, 1)
def log_s(epsilon):
    return -0.5 * tf.log(2*tf.constant(math.pi)) -0.5 * epsilon**2

# Transformation and its derivative
def gamma_h(epsilon, alpha,beta):
    """
    Reparameterization for gamma rejection sampler without shape augmentation.
    """
    b = alpha - 1./3.
    c = 1./tf.sqrt(9.*b)
    v = 1.+epsilon*c
    
    return b*(v**3) 
    
def gamma_h_boosted_B1(epsilon, u, alpha):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = 1#u.shape[1]
    K = alpha.shape[1]#(batch_size,K)
    alpha_vec = alpha
    u_pow = tf.pow(u,1./alpha_vec)+1e-10
    gammah = gamma_h(epsilon, alpha+B,1.)
    return u_pow*gammah
    
def gamma_h_boosted(epsilon, u, alpha,model_B):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    #B = u.shape.dims[0] #u has shape of alpha plus one dimension for B
    B = tf.shape(u)[0]
    K = alpha.shape[1]#(batch_size,K)
    r = tf.range(B)
    rm = tf.to_float(tf.reshape(r,[-1,1,1]))#dim Bx1x1
    alpha_vec = tf.reshape(tf.tile(alpha,(B,1)),(model_B,-1,K)) + rm#dim BxBSxK + dim Bx1
    u_pow = tf.pow(u,1./alpha_vec)+1e-10
    gammah = gamma_h(epsilon, alpha+tf.to_float(B),1.)
    return tf.reduce_prod(u_pow,axis=0)*gammah
    


    
def gamma_grad_h(epsilon, alpha):
    """
    Gradient of reparameterization without shape augmentation.
    """
    b = alpha - 1./3.
    c = 1./tf.sqrt(9.*b)
    v = 1.+epsilon*c
    
    return v**3 - 13.5*epsilon*b*(v**2)*(c**3)
    
def dh(epsilon, alpha, beta):
    return (alpha - 1./3) * 3./tf.sqrt(9*alpha - 3.) * \
           (1+epsilon/tf.sqrt(9*alpha-3))**2 / beta

# Log density of proposal r(z) = s(epsilon) * |dh/depsilon|^{-1}
def log_r(epsilon, alpha, beta): 
    return -tf.log(dh(epsilon, alpha, beta)) + log_s(epsilon)
    
# Density of the accepted value of epsilon 
# (this is just a change of variables too)
def log_pi(eps,alpha):
    beta=1.
    logq=log_q(gamma_h(eps, alpha, beta), alpha, beta)#does not have to be boosted
    return log_s(eps) + \
           logq - \
           log_r(eps, alpha, beta)


def gamma_grad_logr(epsilon, alpha):
    """
    Gradient of log-proposal.
    """
    b = alpha - 1./3.
    c = 1./tf.sqrt(9.*b)
    v = 1.+epsilon*c
    
    return -0.5/b + 9.*epsilon*(c**3)/v
    
def gamma_grad_logq(epsilon, alpha):
    """
    Gradient of log-Gamma at proposed value.
    """
    h_val = gamma_h(epsilon, alpha)
    h_der = gamma_grad_h(epsilon, alpha)
    
    return tf.log(h_val) + (alpha-1.)*h_der/h_val - h_der - tf.digamma(alpha)

def gamma_correction(epsilon, alpha):
    """
    Correction term grad (log q - log r)
    """
    return gamma_grad_logq(epsilon, alpha) - gamma_grad_logr(epsilon,alpha)

def calc_epsilon(gamma,alpha):
    return tf.sqrt(9.*alpha-3.)*(tf.pow(gamma/(alpha-1./3.),1./3.)-1.)

def conditional_sample_gaussian(m, v):
    sample = tf.random.normal(shape=tf.shape(m))
    z = m + (tf.sqrt(v) * sample)
    return z

def train(sess, model, 
          train_url, 
          test_url,train_label_url,test_label_url, label_voc_path,
          batch_size, 
          vocab_size,
          analytical,n_topic,y_dim,
          alternate_epochs=1,#10
          lexicon=[],
          result_file='test.txt',
          B=1,
          warm_up_period=100,
          output_dir='model/'):
  """train nvdm model."""
  train_set, train_count, train_labels = utils.data_set(train_url,train_label_url,label_voc_path)
  test_set, test_count, test_labels = utils.data_set(test_url,test_label_url,label_voc_path)
  # hold-out development dataset
  # print(train_labels[0])
  train_size=len(train_set)
  validation_size=int(train_size*0.1)
  dev_set = train_set[:validation_size]
  dev_count = train_count[:validation_size]
  dev_labels = train_labels[:validation_size]
  train_set = train_set[validation_size:]
  train_count = train_count[validation_size:]
  train_labels = train_labels[validation_size:]
  dev_labels = np.zeros_like(dev_labels)
  test_labels = np.zeros_like(test_labels)
  optimize_jointly = True
  dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
  test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)
  warm_up = 0
  min_alpha = 0.00001#
  curr_B=B

  best_print_ana_ppx=1e50
  early_stopping_iters=30
  no_improvement_iters=0
  stopped=False
  epoch=-1
  #for epoch in range(training_epochs):
  while not stopped:
    epoch+=1
    train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
    if warm_up<1.:
      warm_up += 1./warm_up_period
    else:
      warm_up=1.
   
    #-------------------------------
    # train
    #for switch in range(0, 2):
    if optimize_jointly:
      optim = model.optim_all
      print_mode = 'updating encoder and decoder'
    elif switch == 0:
      optim = model.optim_dec
      print_mode = 'updating decoder'
    else:
      optim = model.optim_enc
      print_mode = 'updating encoder'
    # print(alternate_epochs)
    # exit()
    for i in range(alternate_epochs):
      loss_sum = 0.0
      ana_loss_sum = 0.0
      ppx_sum = 0.0
      kld_sum_train = 0.0
      ana_kld_sum_train = 0.0
      word_count = 0
      doc_count = 0
      recon_sum=0.0
      for idx_batch in train_batches:
        data_batch, label_batch, count_batch, mask = utils.fetch_data(
        train_set, train_labels, train_count, idx_batch, vocab_size)
        input_feed = {model.x.name: data_batch, model.label.name: label_batch, model.mask.name: mask,model.keep_prob.name: 0.75,model.warm_up.name: warm_up,model.min_alpha.name:min_alpha,model.B.name: curr_B,model.dataset_flag.name: 0}
        _, (loss,recon, kld_train,ana_loss,ana_kld_train) = sess.run((optim, 
                                    [model.true_objective, model.recons_loss, model.kld,model.analytical_objective,model.analytical_kld]),
                                    input_feed)
        # print(loss.shape)
        # print(recon.shape)
        # print(kld_train.shape)
        # exit()
        loss_sum += np.sum(loss)
        # print(loss)
        ana_loss_sum += np.sum(ana_loss)
        kld_sum_train += np.sum(kld_train) / np.sum(mask) 
        ana_kld_sum_train += np.sum(ana_kld_train) / np.sum(mask)
        word_count += np.sum(count_batch)
        # to avoid nan error
        # count_batch = np.add(count_batch, 1e-12)
        # per document loss
        ppx_sum += np.sum(np.nan_to_num(np.divide(loss, count_batch), nan=0, posinf=0, neginf=0)) 
        doc_count += np.sum(mask)
        recon_sum+=np.sum(recon)
      dec_vars = utils.variable_parser(tf.trainable_variables(), 'decoder')
      trainable_variables = tf.trainable_variables()
      # print(dec_vars)
      # exit()
      # for var in trainable_variables:
      #   print('Name: {}, Shape: {}'.format(var.name, var.shape))
        # enc_vars = utils.variable_parser(tf.trainable_variables())
        # print(enc_vars)
      # exit()
      # dag_weights = model.dag.get_weights()
      # print(dag_weights)
      # exit()
      phi = dec_vars[0]
      # phi = dec_vars[0]
      # print(phi)
      # exit()
      phi_causal = dec_vars[-2]
        # print('phi')
        # print(phi)
        # exit()
      print_loss = recon_sum/len(train_batches)
      # dec_vars = utils.variable_parser(tf.trainable_variables(), 'decoder')
      # phi = dec_vars[0]
      # print('phi')
      # print(phi)
      # exit()
      phi = sess.run(phi)
      phi_causal = sess.run(phi_causal)
      # print(dec_vars[-6])
      matrix_A = sess.run(dec_vars[2])
      # matrix_A = sess.run(trainable_variables[0])
      # print(dec_vars[0],dec_vars[-2],dec_vars[2])
      # exit()
      
      # print_ppx = np.exp(loss_sum / word_count)
      # print_ana_ppx = np.exp(ana_loss_sum / word_count)
      # print_ppx_perdoc = np.exp(ppx_sum / doc_count)
      print_ppx = np.exp(loss_sum*0.5 / word_count)
      print_ana_ppx = np.exp(ana_loss_sum*0.5 / word_count)
      print_ppx_perdoc = np.exp(ppx_sum*0.5 / doc_count)
      print_kld_train = kld_sum_train/len(train_batches)
      print_ana_kld_train = ana_kld_sum_train/len(train_batches)
      print('| Epoch train: {:d} |'.format(epoch+1), 
               print_mode, '{:d}'.format(i),
               '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
               '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
               '| KLD: {:.5}'.format(print_kld_train),
               '| Loss: {:.5}'.format(print_loss),
               '| ppx anal.: {:.5f}'.format(print_ana_ppx),
               '|KLD anal.: {:.5f}'.format(print_ana_kld_train))
     
    
    #-------------------------------
    # dev
    loss_sum = 0.0
    kld_sum_dev = 0.0
    ppx_sum = 0.0
    word_count = 0
    doc_count = 0
    recon_sum=0.0
    print_ana_ppx = 0.0
    ana_loss_sum = 0.0
    for idx_batch in dev_batches:
      data_batch, label_batch, count_batch, mask = utils.fetch_data(
          dev_set, dev_labels, dev_count, idx_batch, vocab_size)

      input_feed = {model.x.name: data_batch, model.label.name: np.zeros((len(label_batch), len(label_batch[0]))), model.mask.name: mask,model.keep_prob.name: 1.0,model.warm_up.name: 1.0,model.min_alpha.name:min_alpha,model.B.name: B,model.dataset_flag.name: 1}
      loss,recon, kld_dev,ana_kld,ana_loss = sess.run([model.objective, model.recons_loss,model.kld, model.analytical_kld,model.analytical_objective],
                           input_feed)
      loss_sum += np.sum(loss)
      ana_loss_sum += np.sum(ana_loss)
      kld_sum_dev += np.sum(kld_dev) / np.sum(mask)  
      word_count += np.sum(count_batch)
      # count_batch = np.add(count_batch, 1e-12)
      # ppx_sum += np.sum(np.divide(loss, count_batch))
      # count_batch = np.add(count_batch, 1e-12)
      # per document loss
      ppx_sum += np.sum(np.nan_to_num(np.divide(loss, count_batch), nan=0, posinf=0, neginf=0)) 
      doc_count += np.sum(mask) 
      recon_sum+=np.sum(recon)
    print_ana_ppx = np.exp(ana_loss_sum*0.5 / word_count)
    print_ppx = np.exp(loss_sum*0.5 / word_count)
    print_ppx_perdoc = np.exp(ppx_sum*0.5 / doc_count)
    print_kld_dev = kld_sum_dev/len(dev_batches)
    print_loss = recon_sum/len(dev_batches)
    print('| Epoch dev: {:d} |'.format(epoch+1), 
           '| Perplexity: {:.9f}'.format(print_ppx),
           '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
           '| KLD: {:.5}'.format(print_kld_dev)  ,
           '| Loss: {:.5}'.format(print_loss))
    if print_ppx<best_print_ana_ppx:
      no_improvement_iters=0
      best_print_ana_ppx=print_ppx
      #check on validation set, if ppx better-> save improved model
      
      tf.train.Saver().save(sess, output_dir)
      # if epoch>-2: 
      #   utils.print_top_words(phi,phi_causal,matrix_A, lexicon,n_topic,result_file=None,output_dir=output_dir)
      # #-------------------------------
      # # test
      
      # if FLAGS.test:
      #   test_theta=np.zeros(shape=(1,n_topic))
      #   test_theta_causal=np.zeros(shape=(1,n_topic+y_dim))
      #   loss_sum = 0.0
      #   kld_sum_test = 0.0
      #   ppx_sum = 0.0
      #   word_count = 0
      #   doc_count = 0
      #   recon_sum = 0.0
      #   ana_loss_sum = 0.0
      #   ana_kld_sum_test = 0.0
      #   for idx_batch in test_batches:
      #     data_batch, label_batch, count_batch, mask = utils.fetch_data(
      #       test_set, test_labels, test_count, idx_batch, vocab_size)
      #     input_feed = {model.x.name: data_batch, model.label.name: np.zeros((len(label_batch), len(label_batch[0]))), model.mask.name: mask,model.keep_prob.name: 1.0,model.warm_up.name: 1.0,model.min_alpha.name:min_alpha,model.B.name: B,model.dataset_flag.name: 1}
      #     loss, recon,kld_test,ana_loss,ana_kld_test,batch_theta,batch_theta_causal = sess.run([model.true_objective, model.recons_loss,model.kld,model.analytical_objective,model.analytical_kld,model.doc_vec,model.doc_vec_causal],
      #                        input_feed)
      #     test_theta=np.concatenate((test_theta,batch_theta),axis=0)
      #     test_theta_causal=np.concatenate((test_theta_causal,batch_theta_causal),axis=0)
      #     loss_sum += np.sum(loss)
      #     kld_sum_test += np.sum(kld_test)/np.sum(mask) 
      #     ana_loss_sum += np.sum(ana_loss)
      #     ana_kld_sum_test += np.sum(ana_kld_test) / np.sum(mask)
      #     word_count += np.sum(count_batch)
      #         # count_batch = np.add(count_batch, 1e-12)
      #         # ppx_sum += np.sum(np.divide(loss, count_batch))
      #         # count_batch = np.add(count_batch, 1e-12)
      #         # per document loss
      #     ppx_sum += np.sum(np.nan_to_num(np.divide(loss, count_batch), nan=0, posinf=0, neginf=0)) 
      #     doc_count += np.sum(mask) 
      #     recon_sum+=np.sum(recon)
      #   print_loss = recon_sum/len(test_batches)
      #   print_ppx = np.exp(loss_sum / word_count)
      #   print_ppx_perdoc = np.exp(ppx_sum / doc_count)
      #   print_kld_test = kld_sum_test/len(test_batches)
      #   print_ana_ppx = np.exp(ana_loss_sum / word_count)
      #   print_ana_kld_test = ana_kld_sum_test/len(train_batches)
      #   print('| Epoch test: {:d} |'.format(epoch+1), 
      #        '| Perplexity: {:.9f}'.format(print_ppx),
      #        '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
      #        '| KLD: {:.5}'.format(print_kld_test),
      #        '| Loss: {:.5}'.format(print_loss),
      #        '| ppx anal.: {:.5f}'.format(print_ana_ppx),
      #          '|KLD anal.: {:.5f}'.format(print_ana_kld_test))
      #   np.savetxt(output_dir+"test_theta.txt", test_theta[1:len(test_set)+1])
      #   np.savetxt(output_dir+"test_theta_causal.txt", test_theta_causal[1:len(test_set)+1])
      #   if stopped:
      #     #only do it once in the end
      #     print('calculate topic coherence (might take a few minutes)')
      #     coherence=utils.topic_coherence(test_set,phi, lexicon)
      #     print('topic coherence',str(coherence))
      #   # exit()
      
    else:
      no_improvement_iters+=1
      print('no_improvement_iters',no_improvement_iters,'best ppx',best_print_ana_ppx)
      if no_improvement_iters>=early_stopping_iters:
          #if model has not improved for 30 iterations, stop training
          ###########STOP TRAINING############
          stopped=True
          print('stop training after',epoch,'iterations,no_improvement_iters',no_improvement_iters)
          ###########LOAD BEST MODEL##########
          print('load stored model')
          tf.train.Saver().restore(sess,output_dir)
          utils.print_top_words(phi,phi_causal,matrix_A, lexicon,n_topic,result_file=None,output_dir=output_dir)
          #-------------------------------
          # test
          
          if FLAGS.test:
            test_theta=np.zeros(shape=(1,n_topic))
            test_theta_causal=np.zeros(shape=(1,n_topic+y_dim))
            loss_sum = 0.0
            kld_sum_test = 0.0
            ppx_sum = 0.0
            word_count = 0
            doc_count = 0
            recon_sum = 0.0
            ana_loss_sum = 0.0
            ana_kld_sum_test = 0.0
            for idx_batch in test_batches:
              data_batch, label_batch, count_batch, mask = utils.fetch_data(
                test_set, test_labels, test_count, idx_batch, vocab_size)
              input_feed = {model.x.name: data_batch, model.label.name: np.zeros((len(label_batch), len(label_batch[0]))), model.mask.name: mask,model.keep_prob.name: 1.0,model.warm_up.name: 1.0,model.min_alpha.name:min_alpha,model.B.name: B,model.dataset_flag.name: 1}
              loss, recon,kld_test,ana_loss,ana_kld_test,batch_theta,batch_theta_causal = sess.run([model.objective, model.recons_loss,model.kld,model.analytical_objective,model.analytical_kld,model.doc_vec,model.doc_vec_causal],
                             input_feed)
              test_theta=np.concatenate((test_theta,batch_theta),axis=0)
              test_theta_causal=np.concatenate((test_theta_causal,batch_theta_causal),axis=0)
              loss_sum += np.sum(loss)
              kld_sum_test += np.sum(kld_test)/np.sum(mask) 
              ana_loss_sum += np.sum(ana_loss)
              ana_kld_sum_test += np.sum(ana_kld_test) / np.sum(mask)
              word_count += np.sum(count_batch)
              # count_batch = np.add(count_batch, 1e-12)
              # ppx_sum += np.sum(np.divide(loss, count_batch))
              # count_batch = np.add(count_batch, 1e-12)
              # per document loss
              ppx_sum += np.sum(np.nan_to_num(np.divide(loss, count_batch), nan=0, posinf=0, neginf=0)) 
              doc_count += np.sum(mask) 
              recon_sum+=np.sum(recon)
            print_loss = recon_sum/len(test_batches)
            print_ppx = np.exp(loss_sum*0.5 / word_count)
            print_ppx_perdoc = np.exp(ppx_sum*0.5 / doc_count)
            print_kld_test = kld_sum_test/len(test_batches)
            print_ana_ppx = np.exp(ana_loss_sum*0.5 / word_count)
            print_ana_kld_test = ana_kld_sum_test/len(train_batches)
            print('| Epoch test: {:d} |'.format(epoch+1), 
             '| Perplexity: {:.9f}'.format(print_ppx),
             '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
             '| KLD: {:.5}'.format(print_kld_test),
             '| Loss: {:.5}'.format(print_loss),
             '| ppx anal.: {:.5f}'.format(print_ana_ppx),
               '|KLD anal.: {:.5f}'.format(print_ana_kld_test))
            np.savetxt(output_dir+"test_theta.txt", test_theta[1:len(test_set)+1])
            np.savetxt(output_dir+"test_theta_causal.txt", test_theta_causal[1:len(test_set)+1])
            if stopped:
              #only do it once in the end
              print('calculate topic coherence (might take a few minutes)')
              coherence=utils.topic_coherence(test_set,phi, lexicon)
              print('topic coherence',str(coherence))

      

    # #-------------------------------
    # # test
    # if FLAGS.test:
      
    #   loss_sum = 0.0
    #   kld_sum_test = 0.0
    #   ppx_sum = 0.0
    #   word_count = 0
    #   doc_count = 0
    #   recon_sum = 0.0
    #   ana_loss_sum = 0.0
    #   ana_kld_sum_test = 0.0
    #   for idx_batch in test_batches:
    #     data_batch, count_batch, mask = utils.fetch_data(
    #       test_set, test_count, idx_batch, vocab_size)
    #     input_feed = {model.x.name: data_batch, model.mask.name: mask,model.keep_prob.name: 1.0,model.warm_up.name: 1.0,model.min_alpha.name:min_alpha,model.B.name: B}
    #     loss, recon,kld_test,ana_loss,ana_kld_test,theta = sess.run([model.objective, model.recons_loss,model.kld,model.analytical_objective,model.analytical_kld,model.doc_vec],
    #                          input_feed)
    #     print(theta)
    #     exit()
    #     loss_sum += np.sum(loss)
    #     kld_sum_test += np.sum(kld_test)/np.sum(mask) 
    #     ana_loss_sum += np.sum(ana_loss)
    #     ana_kld_sum_test += np.sum(ana_kld_test) / np.sum(mask)
    #     word_count += np.sum(count_batch)
    #     count_batch = np.add(count_batch, 1e-12)
    #     ppx_sum += np.sum(np.divide(loss, count_batch))
    #     doc_count += np.sum(mask) 
    #     recon_sum+=np.sum(recon)
    #   print_loss = recon_sum/len(test_batches)
    #   print_ppx = np.exp(loss_sum / word_count)
    #   print_ppx_perdoc = np.exp(ppx_sum / doc_count)
    #   print_kld_test = kld_sum_test/len(test_batches)
    #   print_ana_ppx = np.exp(ana_loss_sum / word_count)
    #   print_ana_kld_test = ana_kld_sum_test/len(train_batches)
    #   print('| Epoch test: {:d} |'.format(epoch+1), 
    #          '| Perplexity: {:.9f}'.format(print_ppx),
    #          '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
    #          '| KLD: {:.5}'.format(print_kld_test),
    #          '| Loss: {:.5}'.format(print_loss),
    #          '| ppx anal.: {:.5f}'.format(print_ana_ppx),
    #            '|KLD anal.: {:.5f}'.format(print_ana_kld_test)) 
      # if stopped:
      #   #only do it once in the end
      #   print('calculate topic coherence (might take a few minutes)')
      #   coherence=utils.topic_coherence(test_set,phi, lexicon)
      #   print('topic coherence',str(coherence))
     


  
  
def myrelu(features):
    return tf.maximum(features, 0.0)

def parseArgs():
    #get line from config file
    args = sys.argv
    linum = int(args[1])
    argstring=''
    configname = 'tfconfig'
    with open(configname,'r') as rf:
        for i,line in enumerate(rf):
            #print i,line
            argstring = line
            if i+1==linum:
                print(line)
                break
    argparser = argparse.ArgumentParser()
    #define arguments
    argparser.add_argument('--adam_beta1',default=0.9, type=float)
    argparser.add_argument('--adam_beta2',default=0.999, type=float)
    argparser.add_argument('--learning_rate',default=1e-3, type=float)
    argparser.add_argument('--dir_prior',default=0.1, type=float)
    argparser.add_argument('--z2_dim',default=1, type=int)
    argparser.add_argument('--B',default=1, type=int)
    argparser.add_argument('--n_topic',default=50, type=int)
    argparser.add_argument('--n_sample',default=1, type=int)
    argparser.add_argument('--warm_up_period',default=100, type=int)
    argparser.add_argument('--nocorrection',action="store_true")
    argparser.add_argument('--data_dir',default='data/20news', type=str)
    argparser.add_argument('--output_dir',default='model', type=str)
    return argparser.parse_args(argstring.split())

def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
      non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
      non_linearity = tf.nn.sigmoid
    else:
      non_linearity = myrelu
    
    analytical=False
    args = parseArgs()
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    learning_rate = args.learning_rate
    dir_prior = args.dir_prior
    B=args.B
    warm_up_period = args.warm_up_period
    n_sample = args.n_sample
    n_topic = args.n_topic
    lexicon=[]
    output_dir=args.output_dir
    z2_dim=args.z2_dim
    # print('z2_dim = ',z2_dim)
    # if not os.path.exists(output_dir):
    #   os.makedirs(output_dir)
    vocab_path = os.path.join(args.data_dir, 'voc.txt')
    # vocab_path = os.path.join(args.data_dir, 'vocab.new')
    with open(vocab_path,'r') as rf:
        for line in rf:
            word = line.split()[0]
            lexicon.append(word)
    vocab_size=len(lexicon)
    train_url = os.path.join(args.data_dir, 'train.feat')
    test_url = os.path.join(args.data_dir, 'test.feat')

    train_label_url = os.path.join(args.data_dir, 'train_label.txt')
    test_label_url = os.path.join(args.data_dir, 'test_label.txt')
    label_voc_path = os.path.join(args.data_dir, 'label_voc.txt')
    with open(label_voc_path, 'r') as f:
      docs=f.readlines()
    y_dim=len(docs)
  
    nvdm = NVDM(analytical=analytical,
                vocab_size=vocab_size,
                n_hidden=FLAGS.n_hidden,
                n_topic=n_topic, 
                n_sample=n_sample,
                learning_rate=learning_rate, 
                batch_size=FLAGS.batch_size,
                non_linearity=non_linearity,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                B=B,
                dir_prior=dir_prior,
                correction=(not args.nocorrection),
                y_dim=y_dim,
                z2_dim=z2_dim)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    result = sess.run(init)
        
    train(sess, nvdm, train_url, test_url,train_label_url,test_label_url,label_voc_path, FLAGS.batch_size,vocab_size,analytical,n_topic,y_dim,lexicon=lexicon,
                result_file=None,B=B,
                warm_up_period = warm_up_period,output_dir = output_dir)

if __name__ == '__main__':
    tf.app.run()
