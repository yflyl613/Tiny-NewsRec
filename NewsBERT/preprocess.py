import tensorflow as tf
from tqdm import tqdm
import numpy as np
from utils import MODEL_CLASSES


def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value


def read_news_bert(news_path, args, mode='train'):
    news = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name, do_lower_case=True)

    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, _, _, _, _ = splited
            update_dict(news_index, doc_id)

            title = title.lower()
            title = tokenizer(title, max_length=args.num_words_title,
                              pad_to_max_length=True, truncation=True)

            update_dict(news, doc_id, [title, category, subcategory])
            if mode == 'train':
                update_dict(category_dict, category)
                update_dict(subcategory_dict, subcategory)

    if mode == 'train':
        return news, news_index, category_dict, subcategory_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'


def get_doc_input_bert(news, news_index, category_dict, subcategory_dict, args):
    news_num = len(news) + 1

    news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
    news_title_attmask = np.zeros(
        (news_num, args.num_words_title), dtype='int32')
    news_category = np.zeros(news_num, dtype='int32')
    news_subcategory = np.zeros(news_num, dtype='int32')

    for key in tqdm(news):
        title, category, subcategory = news[key]
        doc_index = news_index[key]

        news_title[doc_index] = title['input_ids']
        news_title_attmask[doc_index] = title['attention_mask']
        news_category[doc_index] = category_dict[category] if category in category_dict else 0
        news_subcategory[doc_index] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0

    return news_title, news_title_attmask, news_category, news_subcategory
