import stanza
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DependencyParsingMatrix(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.pipline = stanza.Pipeline('en', download_method=None, processors='tokenize, mwt, pos, lemma, depparse')
        self.max_len = 10

    def __getitem__(self, item):
        """
        分批次执行数据集不一次性全部处理导致数据处理过慢影响模型训练
        :param item: 每一个batch数据
        :return: 每个batch的稀疏矩阵
        """
        # 获取dependency parsing 的结果(deprel, head, id)
        dep_outputs = self.get_output_dep_parsing(self.dataset[item])
        new_dep_outputs = dependency2inx(dep_outputs)
        head = get_head_list(new_dep_outputs)
        adj = head2adj(head, self.max_len)
        return adj

    def __len__(self):
        return len(self.dataset)

    def get_output_dep_parsing(self, text):
        """
        :param text: 原始数据集
        :return:  返回department parsing
        """
        doc = self.pipline(text)
        print(*[
            f'id:{word.id}\t'
            f'word:{word.text}\t'
            f'head id:{word.head}\t'
            f'head:{sent.words[word.head - 1].text if word.head > 0 else "root"}\t'
            f'deprel:{word.deprel} '
            for sent in doc.sentences for word in sent.words], sep='\n')
        words = []
        for s in doc.sentences:
            for word in s.words:
                words.append((word.deprel, word.head, word.id))
        return words


def dependency2inx(dep_outputs):
    """查找根结点对应的索引"""
    root_index = []
    for i in range(len(dep_outputs)):
        if dep_outputs[i][0] == 'ROOT':
            root_index.append(i)

    '''修改依存关系三元组'''
    new_dep_outputs = []
    tag = 0
    for i in range(len(dep_outputs)):
        for index in root_index:
            if i + 1 > index:
                tag = index

        if dep_outputs[i][0] == 'ROOT':
            dep_output = (dep_outputs[i][0], dep_outputs[i][1], dep_outputs[i][2] + tag)
        else:
            dep_output = (dep_outputs[i][0], dep_outputs[i][1] + tag, dep_outputs[i][2] + tag)
        new_dep_outputs.append(dep_output)
    return new_dep_outputs


def get_head_list(new_dep_outputs):
    head_list = []
    for i in range(len(new_dep_outputs)):
        for dep_output in new_dep_outputs:
            if dep_output[-1] == i + 1:
                head_list.append(int(dep_output[1]))
    return head_list


def head2adj(head, max_sent_len):
    ret = np.identity(max_sent_len)
    for i in range(len(head)):
        j = head[i]
        if j != 0:
            if i <= max_sent_len - 1 and j <= max_sent_len - 1:
                ret[i, j - 1] = 1
                ret[j - 1, i] = 1
    return ret


if __name__ == '__main__':
    texts = ["my name is chh .", "I know Bob's girlfriend.", "do you love me ?", "I love eating apple.",
             "We poured the milk into the pumpkin mixture which is contained in a bowl ."]
    dependencyParsingMatrix = DependencyParsingMatrix(texts)
    test = DataLoader(dependencyParsingMatrix, batch_size=2, shuffle=False)
    for w in test:
        print(w)
