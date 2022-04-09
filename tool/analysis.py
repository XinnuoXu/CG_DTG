#coding=utf8

from models.neural import CalculateSelfAttention

class Analysis():

    def __init__(self):

        self.self_attn_layer = CalculateSelfAttention()

    def edge_ranking(self, sents_vec, batch.alg, mask_cls):
        self.self_attn_layer(sents_vec, sents_vec, mask_cls)
        print (sents_vec.size(), mask_cls.size())
