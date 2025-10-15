# coding=utf-8
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import collections
import json
import logging

import torch
from losses import cross_softmax

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult", ["id", "logits"])


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def cal_losses(raw_results, id2feature):
    """

    Args:
        raw_results (list[RawResult]):
        id2feature (dict):

    Returns:
        dict:
    """
    with torch.no_grad():
        labels, logits = [], []
        for r in raw_results:
            f = id2feature[r.id]
            labels.append(torch.tensor(f.labels, dtype=torch.float))
            logits.append(r.logits)

        labels = torch.stack(labels)
        logits = torch.stack(logits)

        _losses = dict()
        # _losses['list_mle'] = list_mle(logits, labels).item()  # KLD
        # _losses['list_net'] = list_net(logits, labels).item()  # Likelihood
        # _losses['approx_ndcg'] = approx_ndcg_loss(logits, labels).item()  # ApproxNDCG
        # _losses['rank_net'] = rank_net(logits, labels).item()  # Hinge
        # _losses['hinge'] = pairwise_hinge(logits, labels).item()  # Hinge
        # _losses['lambda'] = lambda_loss(logits, labels).item()  # LambdaRank
        # _losses['CE'] = CE(logits, labels).item()
        # _losses['MSE'] = MSE(logits, labels).item()
        _losses['cross_softmax'] = cross_softmax(logits, labels).item()
    return _losses


def infer_labels(samples, raw_results, id2example, output_file=None, score_file=None):
    """

    Args:
        samples (list[dict]):
        id2example (dict):
        raw_results (list[RawResult]):
        output_file:
        score_file:

    Returns:

    """
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []

    labels = []
    scores = []
    id2logits = dict(raw_results)
    for sample in samples:
        story_id = sample['story_id']
        hyp1 = sample['hyp1']
        hyp2 = sample['hyp2']
        if story_id not in id2example or story_id not in id2logits:
            labels.append(1)
        else:
            story = id2example[story_id]
            logits = id2logits[story_id]
            hyp1_score = logits[story.hyp2idx[hyp1]]
            hyp2_score = logits[story.hyp2idx[hyp2]]

            # _hyp1_score = torch.sum((hyp1_score + 1) ** 2)
            # _hyp2_score = torch.sum((hyp2_score + 1) ** 2)

            # if _hyp1_score > _hyp2_score:
            #     pos_x.append(hyp1_score[0])
            #     pos_y.append(hyp1_score[1])
            #     neg_x.append(hyp2_score[0])
            #     neg_y.append(hyp2_score[1])
            # else:
            #     pos_x.append(hyp2_score[0])
            #     pos_y.append(hyp2_score[1])
            #     neg_x.append(hyp1_score[0])
            #     neg_y.append(hyp1_score[1])
            labels.append(1 if hyp1_score > hyp2_score else 2)
            scores.append((hyp1_score, hyp2_score))

    if output_file is not None:
        with open(output_file, 'w') as f:
            for label in labels:
                f.write('%d\n' % label)

    if score_file is not None:
        with open(score_file, 'w') as f:
            for s1, s2 in scores:
                f.write('%f, %f\n' % (s1, s2))
    # plot(pos_x, pos_y, neg_x, neg_y)
    return labels


def write_results(comment, metrics, losses, filename):
    with open(filename, 'w') as f:
        if comment:
            f.write(comment + '\n\n')
        f.write('accuracy: %f\n\n' % (metrics['accuracy'] * 100))
        f.write(json.dumps(metrics, ensure_ascii=False) + '\n\n')
        f.write(json.dumps(losses, ensure_ascii=False) + '\n')

# def plot(posx, posy, negx, negy):
#
#     plt.scatter(posx, posy, c='c', edgecolors='r')
#     plt.scatter(negx, negy, c='r', edgecolors='r')
#     plt.show()
