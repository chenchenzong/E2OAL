import numpy as np
import torch
import numpy as np
from utils import lab_conv
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import math

def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, knownclass):
    model.eval()
    queryIndex = []
    labelArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((queryIndex, labelArr)).T
    np.random.shuffle(tmp_data)
    tmp_data = tmp_data.T
    queryIndex = tmp_data[0][:args.query_batch]
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def find_threshold(probs, uncerts, target_sum=900, top_k=1500):
    probs = np.array(probs)
    uncerts = np.array(uncerts)

    sorted_indices = np.argsort(-probs)
    n = len(probs)
    if target_sum <= 0:
        return n
    for k in range(top_k, n + 1):
        top_k_indices = sorted_indices[:k] 
        selected_uncerts = uncerts[top_k_indices]
        selected_probs = probs[top_k_indices]

        sub_indices = np.argsort(-selected_uncerts)[:top_k]
        chosen_probs = selected_probs[sub_indices]

        prob_sum = np.sum(chosen_probs)

        if int(prob_sum) <= target_sum:
            return k
    
    return n

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p + eps
    q = q + eps

    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)

    m = 0.5 * (p + q)

    kl_pm = F.kl_div(torch.log(p), m, reduction='none').sum(dim=1)
    kl_qm = F.kl_div(torch.log(q), m, reduction='none').sum(dim=1)

    js = 0.5 * (kl_pm + kl_qm)
    return js

def e2oal_sampling(args, unlabeledloader, trainloader_ID_w_OOD, Len_labeled_ind_train, target_model, knownclass):
    target_model.eval()
    queryIndex = []
    maxlogitArr = []
    entropyArr = []
    labelArr = []
    predArr = []
    temp_maxlogitArr = []
    precision, recall = 0, 0
    with torch.no_grad():
        for batch_idx, (index, (data, labels)) in enumerate(trainloader_ID_w_OOD):
            labels = lab_conv(knownclass, labels)
            data = data.cuda()
            _, outputs, _ = target_model(data)

            if outputs.shape[1] == len(knownclass):
                maxlogit, _ = outputs.max(1)
            else:
                maxlogit1, _ = outputs[:,:len(knownclass)].max(1)
                maxlogit2, _ = outputs[:,len(knownclass):].max(1)
                maxlogit = maxlogit1 - maxlogit2
            temp_maxlogitArr += list(maxlogit.cpu().data.numpy())

        for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
            data = data.cuda()
            base_outputs, outputs, _ = target_model(data)

            if outputs.shape[1] == len(knownclass):
                maxlogit, pred = outputs.max(1)
            else:
                maxlogit1, _ = outputs[:,:len(knownclass)].max(1)
                maxlogit2, _ = outputs[:,len(knownclass):].max(1)
                maxlogit = maxlogit1 - maxlogit2
                _, pred = outputs.max(1)

            labels = lab_conv(knownclass, labels)
            queryIndex += list(index)
            maxlogitArr += list(maxlogit.cpu().data.numpy())
            temp_maxlogitArr += list(maxlogit.cpu().data.numpy())
            labelArr += list(np.array(labels.cpu().data))
            predArr += list(np.array(pred.cpu().data))

            probs = torch.softmax(base_outputs, dim=1)
            n, c = probs.shape
            max_indices = torch.argmax(probs, dim=1)                   
            one_hot = F.one_hot(max_indices, num_classes=c).float()
            uniform = torch.full_like(probs, fill_value=1.0 / c) 
            
            kl_onehot = js_divergence(probs, one_hot)
            kl_uniform = js_divergence(probs, uniform)
            U = kl_uniform * kl_onehot
            
            entropyArr += list(U.cpu().data.numpy())

    input_temp_maxlogitArr = np.asarray(temp_maxlogitArr).reshape(-1, 1)
    input_maxlogitArr = np.asarray(maxlogitArr).reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_temp_maxlogitArr)
    prob = gmm.predict_proba(input_maxlogitArr) 
    prob1 = prob[:, gmm.means_.argmax()]

    length = find_threshold(prob1, entropyArr, target_sum=args.temp_target_precision*args.query_batch, top_k=args.query_batch)

    queryIndex = torch.tensor(queryIndex)
    labelArr = torch.tensor(labelArr)
    predArr = torch.tensor(predArr)
    cleanArr = torch.tensor(maxlogitArr)
    infoArr = torch.tensor(entropyArr)
    num_per_class = int(math.ceil(args.query_batch/args.known_class))
    k1_per_class = int(math.ceil(length/args.known_class))

    sel_queryIndex, sel_queryLabelArr = [], []
    for i in range(args.known_class):
        temp_queryIndex = queryIndex[predArr == i]
        temp_labelArr = labelArr[predArr == i]
        temp_cleanArr = cleanArr[predArr == i]
        temp_infoArr = infoArr[predArr == i]

        _, indices = torch.topk(temp_cleanArr.view(-1), min(len(temp_cleanArr), k1_per_class))
        temp_queryIndex = temp_queryIndex[indices]
        temp_labelArr = temp_labelArr[indices]
        temp_infoArr = temp_infoArr[indices]

        tmp_data = np.vstack((temp_queryIndex.numpy(), temp_labelArr.numpy()))
        tmp_data = np.vstack((tmp_data, temp_infoArr.numpy())).T
        tmp_data = tmp_data[(-tmp_data[:,2]).argsort()]
        tmp_data = tmp_data.T
        sel_queryIndex += list(tmp_data[0][:num_per_class]) 
        sel_queryLabelArr += list(tmp_data[1][:num_per_class]) 

    queryIndex = np.asarray(sel_queryIndex).astype(np.int32)
    queryLabelArr = np.asarray(sel_queryLabelArr)

    idx = torch.randperm(len(sel_queryIndex))
    queryIndex = queryIndex[idx][:args.query_batch]
    queryLabelArr = queryLabelArr[idx][:args.query_batch]

    labelArr = labelArr.numpy()

    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall, length
