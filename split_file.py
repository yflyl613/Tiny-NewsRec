import random
from tqdm import tqdm
import tensorflow as tf


def get_sample(all_element, num_sample):
    if num_sample > len(all_element):
        return random.sample(all_element * (num_sample // len(all_element) + 1), num_sample)
    else:
        return random.sample(all_element, num_sample)


N = 4
behaviors = []
with open('./MIND/MINDlarge_train/behaviors.tsv') as f:
    for line in tqdm(f):
        iid, uid, time, history, imp = line.strip().split('\t')
        impressions = [x.split('-') for x in imp.split(' ')]
        pos, neg = [], []
        for news_ID, label in impressions:
            if int(label) == 0:
                neg.append(news_ID)
            elif int(label) == 1:
                pos.append(news_ID)
        if len(pos) == 0:
            continue
        for pos_id in pos:
            neg_candidate = get_sample(neg, 4)
            neg_str = ' '.join(neg_candidate)
            new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
            behaviors.append(new_line)

print(len(behaviors))
random.shuffle(behaviors)

split_behaviors = [[] for _ in range(N)]
for i, line in enumerate(behaviors):
    split_behaviors[i % N].append(line)

for i in range(N):
    with tf.io.gfile.GFile(f'./MIND/MINDlarge_train/behaviors_np4_{i}.tsv', 'w') as f:
        for line in split_behaviors[i]:
            f.write(line)
