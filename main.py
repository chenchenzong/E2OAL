import os
import argparse
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from resnet import ResNet50
from resnet_tiny import ResNet50_Tiny
import query_strategies
import datasets
from utils import AverageMeter, get_splits, lab_conv, conv_p, EDL_Loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser("Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tinyimagenet'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--max-query', type=int, default=11)
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--stepsize', type=int, default=60)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
parser.add_argument('--query-strategy', type=str, default='eaoa_sampling', choices=['random', 'e2oal_sampling'])

# model
parser.add_argument('--model', type=str, default='resnet50')
# misc
parser.add_argument('--eval-freq', type=int, default=200)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=20)
parser.add_argument('--init-percent', type=int, default=16)

########
parser.add_argument('--target-precision', type=float, default=0.6)


    

args = parser.parse_args()

import logging

logging_filename = args.dataset + "_strategy_" + args.query_strategy + "_known-class_" + str(args.known_class) \
                    + "_target-precision_" + str(args.target_precision) + "_seed_" + str(args.seed) + ".log"

logging.basicConfig(level=logging.INFO,
                    filename=logging_filename,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

def main():
    seed = 1
    all_accuracies = []
    all_precisions, all_recalls = [], []
        
    knownclass = get_splits(args.dataset, seed, args.known_class)  
    print("Known Classes", knownclass)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
    )

    testloader = dataset.testloader
    unlabeledloader = dataset.unlabeledloader
    trainloader_ID = dataset.trainloader
    trainloader_ID_w_OOD = dataset.trainloader
    invalidList = []
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train

    print("Creating model: {}".format(args.model))
    start_time = time.time()

    Acc, Err, Precision, Recall = {}, {}, {}, {} 
    
    args.temp_target_precision = args.target_precision

    OOD_label = None
    for query in tqdm(range(args.max_query)):
        # Model initialization
        if args.dataset == 'tinyimagenet':
            if query == 0:
                target_model = ResNet50_Tiny(n_class=dataset.num_classes, bn_class=dataset.num_classes)
            else:
                extimate_K, OOD_label = estimateK(args, trainloader_ID_w_OOD, use_gpu, knownclass)
                target_model = ResNet50_Tiny(n_class=extimate_K, bn_class=dataset.num_classes)
        else:
            if query == 0:
                target_model = ResNet50(n_class=dataset.num_classes, bn_class=dataset.num_classes)
            else:
                extimate_K, OOD_label = estimateK(args, trainloader_ID_w_OOD, use_gpu, knownclass)
                target_model = ResNet50(n_class=extimate_K, bn_class=dataset.num_classes)
                
        if use_gpu:
            target_model = nn.DataParallel(target_model).cuda()
            
        criterion_xent = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(target_model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

        if args.stepsize > 0:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

        # Model training 
        for epoch in tqdm(range(args.max_epoch)):

            # Train target model for classifying known classes
            train_model(target_model, criterion_xent, optimizer, trainloader_ID_w_OOD, use_gpu, knownclass, OOD_label, epoch)
                
            if args.stepsize > 0:
                scheduler.step()

            if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
                print("==> Test")
                acc, err = test(target_model, testloader, use_gpu, knownclass)
                bc_acc, bc_err = test_bc(target_model, testloader, use_gpu, knownclass)
                print("Target model | Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
                print("Base Target model | Accuracy (%): {}\t Error rate (%): {}".format(bc_acc, bc_err))
                logging.info("==> Test")
                logging.info("Target model | Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
                logging.info("Base Target model | Accuracy (%): {}\t Error rate (%): {}".format(bc_acc, bc_err))


        # Record results
        acc, err = test(target_model, testloader, use_gpu, knownclass)
        Acc[query], Err[query] = float(acc), float(err) 
        
        queryIndex = []
        if args.query_strategy == "random":
            queryIndex, invalidIndex, Precision[query], Recall[query] = query_strategies.random_sampling(args, unlabeledloader, len(labeled_ind_train), target_model, knownclass)
        elif args.query_strategy == "e2oal_sampling":
            queryIndex, invalidIndex, Precision[query], Recall[query], length = query_strategies.e2oal_sampling(args, unlabeledloader, trainloader_ID_w_OOD, len(labeled_ind_train), target_model, knownclass)
            logging.info("Current length value is " + str(length))

            args.temp_target_precision = min(args.temp_target_precision + (args.target_precision - Precision[query]), 1)
            args.temp_target_precision = max(args.temp_target_precision, 0)
            logging.info("Current temp_target_precision value is " + str(args.temp_target_precision))


        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train = list(set(unlabeled_ind_train)-set(queryIndex))
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)
        invalidList = list(invalidList) + list(invalidIndex)

        print("Query Strategy: "+args.query_strategy+" | Query Budget: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train)))
        logging.info("Query Strategy: "+args.query_strategy+" | Query Budget: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train)))
        dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=list(set(unlabeled_ind_train) - set(invalidList)), labeled_ind_train=labeled_ind_train+invalidList,
        )
        trainloader_ID_w_OOD, unlabeledloader = dataset.trainloader, dataset.unlabeledloader
        testloader = dataset.testloader
        trainloader_ID = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
        ).trainloader
        
    all_accuracies.append(Acc)
    all_precisions.append(Precision)
    all_recalls.append(Recall)
    print("Accuracies", all_accuracies)
    print("Precisions", all_precisions)
    print("Recalls", all_recalls)

    logging.info("Accuracies: %s", all_accuracies)
    logging.info("Precisions: %s", all_precisions)
    logging.info("Recalls: %s", all_recalls)

        
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    logging.info("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train_model(model, criterion_xent, optimizer_model, trainloader_ID_w_OOD, use_gpu, knownclass, OOD_label, epoch):
    model.train()
    losses = AverageMeter()
 
    for batch_idx, (index, (data, labels)) in enumerate(trainloader_ID_w_OOD):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        
        for i in range(len(labels)):
            if labels[i] == len(knownclass):
                labels[i] = OOD_label[index[i]]
                
        base_outputs, outputs, _ = model(data)
        
        loss_xent = criterion_xent(base_outputs[labels<len(knownclass)], labels[labels<len(knownclass)])
        loss_edl = EDL_Loss(outputs.shape[1])(outputs, labels, epoch)
        loss = loss_xent + loss_edl
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        
        losses.update(loss.item(), labels.size(0))

def cluster_acc(y_true, y_pred, known_classes):
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size

    w = np.zeros((y_pred.max()+1, y_true.max()+1), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    
    log_sum_f1 = 0.
    count, unk_count = 0, 0
    for i, j in ind:
        if j < known_classes:
            recall = w[i, j]/((y_true==j).sum())
            precision = w[i, j]/((y_pred==i).sum())
            f1 = torch.tensor(2.*precision*recall/(precision+recall))
            log_sum_f1 += torch.log(f1)
            if (y_true==known_classes).sum() > 0:
                count += (y_pred==i).sum()
                unk_count += w[i, known_classes]
    precision = ((y_true==known_classes).sum()-unk_count)/(y_true.size-count)
    recall = ((y_true==known_classes).sum()-unk_count)/((y_true==known_classes).sum())
    f1 = torch.tensor(2.*precision*recall/(precision+recall))
    log_sum_f1 += torch.log(f1)
    return torch.exp(log_sum_f1/(known_classes+1)), ind
    
def test_kmeans(K, feat_all, labelArr, known_classes):
    kmeans = KMeans(n_clusters=K, random_state=0, init='k-means++').fit(feat_all)
    #kmeans = MiniBatchKMeans(n_clusters=K, batch_size=1024, random_state=0, init='k-means++', n_init=10).fit(feat_all)
    preds = kmeans.labels_
    known_acc, mapping = cluster_acc(labelArr, preds.astype(int), known_classes)
    return known_acc, kmeans, mapping

def ternary_search(left, right, feat_all, labelArr, known_classes):
    count_i = 0
    while right - left > 2:
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        f_m1, _, _ = test_kmeans(m1, feat_all, labelArr, known_classes)
        f_m2, _, _ = test_kmeans(m2, feat_all, labelArr, known_classes)
        if f_m1 < f_m2:
            left = m1
        else:
            right = m2
            
        print(f'Iter {count_i}: m1 {m1}, Acc {f_m1:.4f} | m2 {m2}, Acc {f_m2:.4f} ')
        count_i += 1

    max_val, max_kmeans, max_mapping = test_kmeans(left, feat_all, labelArr, known_classes)
    max_idx = left
    for i in range(left + 1, right + 1):
        f_i, kmeans_i, mapping_i = test_kmeans(i, feat_all, labelArr, known_classes)
        if f_i > max_val:
            max_val = f_i
            max_kmeans = kmeans_i
            max_mapping = mapping_i
            max_idx = i
    return max_idx, max_kmeans, max_mapping

def estimateK(args, trainloader_ID_w_OOD, use_gpu, knownclass):
    if args.dataset == "cifar10":
        CLIP_features = torch.load('./features/CLIP/cifar10_features.pt')
    elif args.dataset == "cifar100":
        CLIP_features = torch.load('./features/CLIP/cifar100_features.pt')
    elif args.dataset == "tinyimagenet":
        CLIP_features = torch.load('./features/CLIP/tinyimagenet_features.pt')
    
    saveIndex = []
    labelArr = []
    for batch_idx, (index, (data, labels)) in enumerate(trainloader_ID_w_OOD):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        saveIndex += index
        labelArr += list(np.array(labels.cpu().data))
    labelArr = np.asarray(labelArr)
    feat_all = CLIP_features[saveIndex]
    feat_all = F.normalize(feat_all, dim=1).cpu().numpy()
    
    big_k = 1000
    small_k = len(knownclass)+1
    best_acc_at_k, max_kmeans, max_mapping = ternary_search(small_k, big_k, feat_all, labelArr, len(knownclass))
    print(f'Best Acc so far at K {best_acc_at_k}')
    
    distances = np.linalg.norm(feat_all[:, np.newaxis] - max_kmeans.cluster_centers_, axis=2)
    probabilities = torch.softmax(torch.tensor(-distances), dim=1)
    
    print(max_mapping)
    ID_list = []
    for i, j in max_mapping:
        if j < len(knownclass):
            ID_list.append(i)
    OOD_list = [x for x in range(best_acc_at_k) if x not in ID_list]

    _, temp_pred_label = torch.max(probabilities, 1)

    unique_values, counts = torch.unique(temp_pred_label[labelArr>=len(knownclass)], return_counts=True)

    for val, count in zip(unique_values, counts):
        print(f"Value {val.item()} appears {count.item()} times.")

    _, pred_label = torch.max(probabilities[:, OOD_list], 1)

    unique_values, counts = torch.unique(pred_label[labelArr>=len(knownclass)], return_counts=True)

    for val, count in zip(unique_values, counts):
        print(f"Value {val.item()} appears {count.item()} times.")
    
    print(pred_label, pred_label.max(), pred_label.min())
    
    return best_acc_at_k, {i: l for i, l in zip(saveIndex, pred_label + len(knownclass))} 
    
 
def test(model, testloader, use_gpu, knownclass):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for index, (data, labels) in testloader:
            labels = lab_conv(knownclass, labels)
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            _, outputs, _ = model(data)
            outputs = outputs[:, :len(knownclass)]
            probs = conv_p(outputs)
            predictions = probs.data.max(1)[1]
            
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    
    return acc, err

def test_bc(model, testloader, use_gpu, knownclass):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for index, (data, labels) in testloader:
            labels = lab_conv(knownclass, labels)
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs, _, _ = model(data)
            outputs = outputs[:, :len(knownclass)]
            probs = F.softmax(outputs, dim=1)
            predictions = probs.data.max(1)[1]
            
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    
    return acc, err


if __name__ == '__main__':
    main()





