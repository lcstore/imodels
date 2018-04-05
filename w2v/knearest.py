# -*- coding: utf-8 -*-

"""

"""
import numpy as np


def load_embedding(emb_path):
    emb_dict = {}
    with open(emb_path) as embf:
        for line in embf:
            unit_arr = line.strip().split(b" ")
            if len(unit_arr) < 200:
                continue
            word = unit_arr[0]
            vector = np.array([float(x) for x in unit_arr[1:]])
            norm_vector = vector * 1.0 / np.linalg.norm(vector)
            emb_dict[word] = norm_vector
    return emb_dict


def cosine(lvector, rvector):
    dot_val = np.dot(lvector.transpose(), rvector)
    lNorm = np.linalg.norm(lvector)
    rNorm = np.linalg.norm(rvector)
    cos_val = dot_val / (lNorm * rNorm)
    return cos_val


def nearby(word, emb_dict, kNum=5):
    if word not in emb_dict:
        print("Not Found word:%s" % (word))
        return
    w_vec = emb_dict[word]
    cos_list = []
    for k, v in emb_dict.items():
        if k == word:
            continue
        cos_val = cosine(w_vec, v)
        cos_list.append({"w": k, "score": cos_val})
    cos_list.sort(key=lambda item: item["score"], reverse=True)
    cos_list = cos_list[0:min(kNum, len(cos_list))]
    return cos_list


def sum_vector(emb_dict, token_arr):
    sum_vec = None
    for token in token_arr:
        if token not in emb_dict:
            continue
        cur_vector = emb_dict[token]
        if sum_vec is None:
            sum_vec = cur_vector
        else:
            sum_vec = sum_vec + cur_vector
    return sum_vec


def main():
    # emb_dict = load_embedding("/apps/docker/tensorflow/google-word2vec/data/content.movie.vec")
    emb_dict = load_embedding("/apps/docker/tensorflow/google-word2vec/data/content.token.vec")
    # emb_dict = load_embedding("/apps/docker/tensorflow/google-word2vec/data/temp.vec")
    print("emb_dict:%s" % (str(len(emb_dict))))
    word = "功夫"
    cos_list = nearby(word, emb_dict, kNum=10)
    if cos_list is not None:
        index = 0
        for item in cos_list:
            index = index + 1
            print("index:%s,word:%s,score:%s" % (str(index), str(item["w"]), str(item["score"])))
    src_title = "唐人 j 探案 2.2018.hc1080p. 国语 中 字 torrent"
    # src_title = "唐人街 探案 2.2018.hcrip.1080p.torrent"
    splitor = b" "
    src_sum_vector = sum_vector(emb_dict, src_title.split(splitor))
    select_lists = ["2015 中国城", "1992 荒 唐人 梦", "2018 唐人街 探案 2", "2018 唐 探 2", "detective chinatown vol 2", "拳脚 刑警 唐人街",
                    "2018 平行 世界 之门 gateway", "2018 动物 世界", "2018 最后 一搏 last full measure"]

    print("src_title:%s" % (src_title))
    title_cos_list = []
    for sitem in select_lists:
        select_sum_vector = sum_vector(emb_dict, sitem.split(splitor))
        if select_sum_vector is None:
            continue
        cos_val = cosine(src_sum_vector, select_sum_vector)
        title_cos_list.append({"title": sitem, "score": cos_val})
    title_cos_list.sort(key=lambda item: item["score"], reverse=True)
    index = 0
    for item in title_cos_list:
        index = index + 1
        print("index:%s,title:%s,score:%s" % (str(index), item["title"], item["score"]))


if __name__ == '__main__':
    main()
