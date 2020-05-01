'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
import torch

# 是否激活cuda
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _dataset
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs, mrrs = [],[], []
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (np.mean(hits), np.mean(ndcgs))
    # Single thread
    for idx in range(len(_testRatings)):
        (hr,ndcg, mrr) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
    return (np.mean(hits), np.mean(ndcgs))

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx][0:999]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int64')
    batch_users, batch_items = torch.LongTensor(users), torch.LongTensor(items)
    tensor_users, tensor_items = batch_users.to(device), batch_items.to(device)
    y_pred = _model(tensor_users, tensor_items) # model predict
    y_pred = y_pred.cpu()
    y_pred = y_pred.detach().numpy()

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = y_pred[i]
    items.pop()
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    mrr = getMRR(ranklist, gtItem)
    # print(hr, ndcg)
    return (hr, ndcg, mrr)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def getMRR(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return float(1.0) / (i+1)
    return 0