"""
This file contains the implementation of the diffusion models
"""
import numpy as np


def coicm(G, neg_seeds, pos_seeds, iter_n):
    """
    the campaign oblivious independent cascade model
    :param G: the networkx graph
    :param neg_seeds: the negative seeds
    :param pos_seeds: the positive seeds
    :param iter_n: the iteration number
    :return: the negative adopters
    """
    time_delay = 0
    impressed = []
    pos_impressed = []
    neg_impressed = []
    neg_front = list(neg_seeds[:])
    pos_front = list(pos_seeds[:])
    time_step = 0
    while len(neg_front):  # | len(pos_front): propogate until negative adopters are exhausted

        impressed.extend(neg_front)
        neg_impressed.extend(neg_front)
        impressed.extend(pos_front)
        pos_impressed.extend(pos_front)
        new_pos_front = []

        if ((time_delay == 0) | (time_step == time_delay)):
            for node in pos_front:
                neighbours = list(G.neighbors(node))
                np.random.seed(iter_n)
                rand_list = np.random.uniform(0, 1, len(neighbours))
                for i in range(len(neighbours)):
                    if rand_list[i] < G[node][neighbours[i]]['positive'] and neighbours[i] not in impressed and \
                            neighbours[i] not in new_pos_front:
                        new_pos_front.append(neighbours[i])

            pos_front = new_pos_front[:]

        new_neg_front = []
        for node in neg_front:
            neighbours = list(G.neighbors(node))
            np.random.seed(iter_n)
            rand_list = np.random.uniform(0, 1, len(neighbours))
            for i in range(len(neighbours)):
                if rand_list[i] < G[node][neighbours[i]]['negative'] and neighbours[i] not in impressed \
                        and neighbours[i] not in new_neg_front and neighbours[i] not in new_pos_front:
                    new_neg_front.append(neighbours[i])

        neg_front = new_neg_front[:]
        time_step = time_step + 1
    return neg_impressed


def icm(G, seeds, iter_n):
    """
    the independent cascade model
    :param G: the networkx graph
    :param seeds: the negative seeds
    :param iter_n: the iteration number
    :return: the negative adopters
    """
    impressed = []
    front = list(seeds[:])

    while front:
        impressed.extend(front)
        new_front = []

        for node in front:
            neighbours = list(G.neighbors(node))
            np.random.seed(iter_n)
            rand_list = np.random.uniform(0, 1, len(neighbours))
            for i in range(len(neighbours)):
                if rand_list[i] < G[node][neighbours[i]]['negative'] and neighbours[i] not in impressed and neighbours[i] not in new_front:
                    new_front.append(neighbours[i])

        front = new_front[:]
    return impressed
