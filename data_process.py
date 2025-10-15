# coding=utf-8
from __future__ import absolute_import, division, print_function

import collections
import csv
from io import open
import json
import logging
import os

import torch
from torch.utils.data import Dataset

from tqdm import tqdm, tqdm_notebook

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

PAD_Y_VAL = -1


class StoryExample(object):

    def __init__(self, _id, obs1, obs2, hypes, labels=None):
        """

        Args:
            _id (str):
            obs1 (str):
            obs2 (str):
            hypes (list[str]):
            labels (list[float]):
        """
        self._id = _id
        self.obs1 = obs1
        self.obs2 = obs2
        self.hypes = hypes
        self.hyp2idx = dict([(hyp, i) for i, hyp in enumerate(hypes)])  # [(hyp1, 0), (hyp2, 1), (hyp3, 2)...]
        self.labels = labels

    @property
    def id(self):
        return self._id

    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)


class StoryFeatures(object):

    def __init__(self, _id, token_ids, token_type_ids, input_mask, labels=None):
        self._id = _id
        self.token_ids = token_ids  # (max_hyp_num, max_seq_len)
        self.token_type_ids = token_type_ids  # (max_hyp_num, max_seq_len)
        self.input_mask = input_mask  # (max_hyp_num, max_seq_num)
        self.labels = labels  # (max_hyp_num,)

    @property
    def id(self):
        return self._id

    @classmethod
    def convert_from_examples(cls, examples, tokenizer,
                              max_hyp_num=22,
                              max_seq_len=512,
                              pad_seg_id=0,
                              mask_padding_with_zero=True):
        """

        Args:
            examples (list[StoryExample]):
            tokenizer (PreTrainedTokenizer):
            max_hyp_num (int):
            max_seq_len (int):
            pad_seg_id (int):
            mask_padding_with_zero (int):

        Returns:
            list[StoryFeatures]:
        """
        # 获取分隔符的token序列
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id     # 填充id
        pad_seq_ids = [pad_id] * max_seq_len
        # features:[
        #               (id,
        #                [hyp1-token_ids,
        #                 hyp2-token_ids,
        #                 hyp3-token_ids,
        #                 ......],
        #                [hyp1-token_type_ids,
        #                 hyp2-token_type_ids,
        #                 hyp3-token_type_ids,
        #                 ......],
        #                [hyp1-input_mask,
        #                 hyp2-input_mask,
        #                 hyp3-input_mask,
        #                 ......],
        #                [1, 0, 1......])
        #            ]
        features = []
        for (e_idx, example) in enumerate(examples):    # 遍历所有样本，train.examples文件
            obs1_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.obs1))   # 得到ob1的token序列
            obs2_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.obs2))   # 得到ob2的token序列

            token_ids = []
            token_type_ids = []
            input_mask = []
            # 生成当前观测样本下，所有假设的token序列
            for hyp in example.hypes:
                if len(token_ids) >= max_hyp_num:
                    break
                hyp_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(hyp))    # 得到hyp的token序列
                # 如果长度过长，对hyp的token序列进行截取
                if len(obs1_token_ids) + len(hyp_token_ids) + len(obs2_token_ids) + 4 > max_seq_len:
                    hyp_token_ids = hyp_token_ids[:max_seq_len - 4 - len(obs1_token_ids) - len(obs2_token_ids)]
                _token_ids = ([cls_id] +
                              obs1_token_ids + [sep_id] +
                              hyp_token_ids + [sep_id] +
                              obs2_token_ids + [sep_id])        # 生成整句话的token序列，即ob1 + hyp + ob2
                _token_type_ids = ([0] * (len(obs1_token_ids) + 2) +
                                   [1] * (len(hyp_token_ids) + 1) +
                                   [0] * (len(obs2_token_ids) + 1))     # 生成区分是哪句话的序列
                _input_mask = [1 if mask_padding_with_zero else 0] * len(_token_ids)    # 全为1的padding序列
                # 如果长度过短，使用[pad]补充
                if len(_token_ids) < max_seq_len:
                    _token_ids += [pad_id] * (max_seq_len - len(_token_ids))
                    _token_type_ids += [pad_seg_id] * (max_seq_len - len(_token_type_ids))
                    _input_mask += [0 if mask_padding_with_zero else 1] * (max_seq_len - len(_input_mask))
                assert len(_token_ids) == len(_token_type_ids) == len(_input_mask) == max_seq_len
                token_ids.append(_token_ids)
                token_type_ids.append(_token_type_ids)
                input_mask.append(_input_mask)
            while len(token_ids) < max_hyp_num:
                token_ids.append(pad_seq_ids)
                token_type_ids.append([0] * max_seq_len)
                input_mask.append([0 if mask_padding_with_zero else 1] * max_seq_len)
            assert len(token_ids) == len(token_type_ids) == len(input_mask) == max_hyp_num

            labels = example.labels
            if labels is not None:
                if len(labels) < max_hyp_num:
                    labels += [PAD_Y_VAL] * (max_hyp_num - len(example.labels))
                elif len(labels) > max_hyp_num:
                    labels = labels[:max_hyp_num]
                assert len(labels) == max_hyp_num

            if e_idx < 5:
                logger.info("*** No. %d example ***", e_idx)
                logger.info("story_id: %s", example.id)
                logger.info("tokens:\n%s",
                            "\n".join([" ".join(tokenizer.convert_ids_to_tokens(r[:40])) for r in token_ids[:3]]))
                logger.info("token_ids:\n%s",
                            "\n".join([" ".join([str(c) for c in r[:40]]) for r in token_ids[:3]]))
                logger.info("token_type_ids:\n%s",
                            "\n".join([" ".join([str(c) for c in r[:40]]) for r in token_type_ids[:3]]))
                logger.info("input_mask:\n%s",
                            "\n".join([" ".join([str(c) for c in r[:40]]) for r in input_mask[:3]]))
                if labels is not None:
                    logger.info("labels: %s", " ".join([str(x) for x in labels]))

            features.append(cls(example.id, token_ids, token_type_ids, input_mask, labels))

        return features


class AlphaNliDataset(Dataset):
    def __init__(self, features, tt_max_hyp_num):
        """

        Args:
            features (list[StoryFeatures]):
            tt_max_hyp_num (int): The maximum number of hypotheses at training time
        """
        self.features = features
        self.tt_max_hyp_num = tt_max_hyp_num

    def __getitem__(self, index):
        f = self.features[index]
        token_ids = torch.tensor(f.token_ids, dtype=torch.long)  # (max_hyp_num, max_seq_len)
        token_type_ids = torch.tensor(f.token_type_ids, dtype=torch.long)  # (max_hyp_num, max_seq_len)
        input_mask = torch.tensor(f.input_mask, dtype=torch.float)  # (max_hyp_num, max_seq_len)
        labels = torch.tensor(f.labels, dtype=torch.float) if f.labels else -1  # (max_hyp_num,)
        f_id = f.id

        # For test and dev set
        if not f.labels or token_ids.size(0) == 2:
            return token_ids, token_type_ids, input_mask, labels, f_id

        real_hyp_num = (labels != -1.).sum().item()
        # For train set, make sure there is at least one positive hypothesis
        while True:
            sampled_idxes = torch.randperm(real_hyp_num)[:self.tt_max_hyp_num]
            if real_hyp_num < self.tt_max_hyp_num:
                sampled_idxes = torch.cat((sampled_idxes, torch.arange(real_hyp_num, self.tt_max_hyp_num)))
            if 1. in labels[sampled_idxes]:
                break
            else:
                logger.debug('retry')

        assert sampled_idxes.size(0) == self.tt_max_hyp_num
        return (token_ids[sampled_idxes],  # (tt_max_hyp_num, max_seq_len)
                token_type_ids[sampled_idxes],  # (tt_max_hyp_num, max_seq_len)
                input_mask[sampled_idxes],  # (tt_max_hyp_num, max_seq_len)
                labels[sampled_idxes],  # (tt_max_hyp_num,)
                f_id)

    def __len__(self):
        return len(self.features)


class AlphaNliProcessor(object):

    def __init__(self, data_dir, tokenizer):
        """

        Args:
            data_dir (str):
            tokenizer (PreTrainedTokenizer):
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
    # 生成train.examples，该文件内容为所有样本的列表，
    # 格式为：id, ob1, ob2, (hyp1,hyp2,hyp3...), [(hyp1, 0), (hyp2, 1), (hyp3, 2)...], [1, 0, 1...]
    def get_examples(self, mode='train', force_preprocess=False):
        """

        Args:
            mode:
            force_preprocess:

        Returns:
            list[StoryExample]:
        """
        logger.info("***** Loading %s examples *****", mode)
        cache_filename = '%s.examples' % mode
        cache_file_path = os.path.join(self.data_dir, cache_filename)
        if not os.path.exists(cache_file_path) or force_preprocess:
            self._preprocess(os.path.join(self.data_dir, cache_filename.replace('examples', 'jsonl')),
                             os.path.join(self.data_dir, '%s-labels.lst' % mode),
                             cache_file_path)   # self._preprocess(train.jsonl, train-labels.lst, train.examples)
        return torch.load(cache_file_path)      # train.examples

    @classmethod
    def get_test_examples(cls, samples):
        logger.info("***** Loading test examples *****")
        stories = collections.defaultdict(dict)
        num_contradictory = 0
        for sample in samples:
            if sample['hyp1'] == sample['hyp2']:
                num_contradictory += 1
            if sample['story_id'] not in stories:
                stories[sample['story_id']]['cnt'] = 1
                stories[sample['story_id']]['obs1'] = sample['obs1']
                stories[sample['story_id']]['obs2'] = sample['obs2']
                stories[sample['story_id']]['hypes'] = []
                stories[sample['story_id']]['hypes'].append(sample['hyp1'])
                stories[sample['story_id']]['hypes'].append(sample['hyp2'])
            else:
                stories[sample['story_id']]['cnt'] += 1
                if not (stories[sample['story_id']]['obs1'] == sample['obs1'] and
                        stories[sample['story_id']]['obs2'] == sample['obs2']):
                    logger.error('Inconsistent story: %s', sample['story_id'])
                stories[sample['story_id']]['hypes'].append(sample['hyp1'])
                stories[sample['story_id']]['hypes'].append(sample['hyp2'])
        logger.info('%d contradictory samples:', num_contradictory)
        examples = []
        for _id, story in stories.items():
            examples.append(StoryExample(_id, story['obs1'], story['obs2'], list(story['hypes'])))
        return examples

    def _preprocess(self, sample_file, label_file, output_file):
        logger.info("***** Pre-processing %s *****" % sample_file)
        stories = collections.defaultdict(dict)     # 新建字典stories 存放所有样本，包括同id样本数量、ob1、ob2、hyp1、hyp2、和标签
        num_duplicate = 0
        for sample in tqdm(self.get_samples(sample_file, label_file)):      # 得到样本与标签一一对应的列表，并对每一个样本进行循坏
            label = sample['label']
            if sample['hyp1'] == sample['hyp2']:
                num_duplicate += 1
                continue
            if sample['story_id'] not in stories:
                stories[sample['story_id']]['cnt'] = 1
                stories[sample['story_id']]['obs1'] = sample['obs1']
                stories[sample['story_id']]['obs2'] = sample['obs2']
                stories[sample['story_id']]['hypes'] = []
                stories[sample['story_id']]['hypes'].append(sample['hyp1'])
                stories[sample['story_id']]['hypes'].append(sample['hyp2'])
                stories[sample['story_id']]['labels'] = []
                stories[sample['story_id']]['labels'].append(1 if label == 1 else 0)
                stories[sample['story_id']]['labels'].append(1 if label == 2 else 0)
            else:
                stories[sample['story_id']]['cnt'] += 1
                assert stories[sample['story_id']]['obs1'] == sample['obs1']    # 检查ob是否相同
                assert stories[sample['story_id']]['obs2'] == sample['obs2']
                stories[sample['story_id']]['hypes'].append(sample['hyp1'])
                stories[sample['story_id']]['hypes'].append(sample['hyp2'])
                stories[sample['story_id']]['labels'].append(1 if label == 1 else 0)
                stories[sample['story_id']]['labels'].append(1 if label == 2 else 0)
        logger.info('%d duplicate hypotheses samples:', num_duplicate)
        examples = []
        for _id, story in stories.items():      # 对stories进行遍历
            examples.append(StoryExample(_id, story['obs1'], story['obs2'],
                                         list(story['hypes']),  # id, ob1, ob2, (hyp1,hyp2..), [1, 0..]
                                         list(story['labels'])))
        torch.save(examples, output_file)
        logger.info("***** Saved to %s *****" % output_file)

    @classmethod
    def get_samples(cls, sample_file, label_file):
        samples = []
        for sample, label in tqdm(zip(cls.read_jsonl(sample_file), cls.read_lst(label_file))):  # 得到样本列表和标签列表并打包对应
            sample['label'] = int(label)
            samples.append(sample)
        return samples      # 返回样本与标签对应的列表

    @classmethod
    def read_json(cls, input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        return data

    @classmethod
    def read_jsonl(cls, input_file):
        """Reads a jsonl file."""
        lines = []
        with open(input_file, "r") as f:
            for line in f:
                dic = json.loads(line)
                lines.append(dic)
        return lines       # 返回样本列表

    @classmethod
    def read_lst(cls, input_file):
        """Reads a lst file."""
        with open(input_file, "r") as f:
            lines = f.readlines()
        return lines       # 返回标签列表

    @classmethod
    def read_tsv(cls, input_file, quote_char=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quote_char)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
