# -*- coding: utf-8 -*-

"""

"""
import argparse
import json
import re
import sys
import time

import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')


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
    if lNorm == 0 or rNorm == 0:
        return 0
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


def sum_vector(emb_dict, token_arr, idf_dict):
    sum_vec = None
    idf_dict = idf_dict or {}
    for token in token_arr:
        if token not in emb_dict:
            # print("emb_dict.miss token:%s" % (token))
            continue
        cur_vector = emb_dict[token]
        if token in idf_dict:
            cur_vector = cur_vector * (1 + idf_dict[token])
        if sum_vec is None:
            sum_vec = cur_vector
        else:
            sum_vec = sum_vec + cur_vector
    return sum_vec


def token_vectors(emb_dict, token_dict, idf_dict):
    """
    change tokens to vector,with idf weight
    :param emb_dict:
    :param token_dict:
    :param idf_dict:
    :return:
    """
    vector_dict = {}
    splitor = b" "
    for k, v in token_dict.items():
        vector = sum_vector(emb_dict, v.split(splitor), idf_dict)
        vector_dict[k] = vector
    return vector_dict


def load_pairs(token_path, splitor=b"="):
    """
    load analysis result to dict.(id=token)
    :param token_path:
    :return:
    """
    token_dict = {}
    with open(token_path) as embf:
        for line in embf:
            unit_arr = line.strip().split(splitor)
            if len(unit_arr) < 2:
                continue
            id = unit_arr[0].strip()
            token = unit_arr[1].strip()
            token_dict[id] = token
    return token_dict


def load_json_dict(token_path):
    """
    load analysis json to dict
    """
    token_dict = {}
    with open(token_path) as embf:
        for line in embf:
            line = line.strip()
            try:
                link_dict = json.loads(line)
            except:
                continue
            if 'content' not in link_dict:
                continue
            id = link_dict['id']
            if id not in token_dict:
                link_contents = []
                link_dict['contents'] = link_contents
                token_dict[id] = link_dict
            elif 'contents' in link_dict:
                link_contents = link_dict['contents']
            link_contents.append(link_dict['content'])
    return token_dict


def count_dict(tokens):
    token_count = {}
    tokens = tokens or []
    for token in tokens:
        if token in token_count:
            token_count[token] = token_count[token] + 1
        else:
            token_count[token] = 1
    return token_count


def tf_dict(token_count):
    tf_map = {}
    if token_count is None:
        return tf_map
    sum = 0
    for k, v in token_count.items():
        sum = sum + v
    for k, v in token_count.items():
        tf_val = float(v) / sum
        tf_map[k] = tf_val
    return tf_map;


def title_vectors(token_unions, token_count, idf_dict):
    tf_map = tf_dict(token_count)
    vectors = []
    for key in token_unions.keys():
        if key not in idf_dict:
            continue
        tf_val = 0
        if key in tf_map:
            tf_val = tf_map[key]
        idf_val = idf_dict[key]
        tfidf = tf_val * idf_val
        # print("key:%s,tf_val:%s,idf_val:%s" % (key, str(tf_val), str(idf_val)))
        vectors.append(tfidf)
    vector = np.array([float(x) for x in vectors])
    norm_val = np.linalg.norm(vector);
    norm_vector = vector
    if norm_val != 0:
        norm_vector = vector * 1.0 / norm_val;
    return norm_vector


def empty_dict(inDict):
    return inDict is None or (len(inDict.keys()) < 1)


def intersection_keys(dict1, dict2):
    return list([x for x in dict1 if x in dict2])


def tfidf_cosine(ltoken_count, rtoken_count, idf_dict):
    if empty_dict(ltoken_count) or empty_dict(rtoken_count):
        return 0
    interKeys = intersection_keys(ltoken_count, rtoken_count)
    if len(interKeys) < 1:
        return 0

    token_unions = {}
    for k, v in ltoken_count.items():
        if k in token_unions:
            token_unions[k] = token_unions[k] + v
        else:
            token_unions[k] = v
    ltoken_vector = title_vectors(token_unions, ltoken_count, idf_dict)
    rtoken_vector = title_vectors(token_unions, rtoken_count, idf_dict)
    return cosine(ltoken_vector, rtoken_vector)


def w2v_knn(emb_dict, link_dict, content_dict, idf_dict):
    link_vectors = token_vectors(emb_dict, link_dict, idf_dict)
    content_vectors = token_vectors(emb_dict, content_dict, idf_dict)
    link_cosines_dict = {}
    for lk, lv in link_vectors.items():
        if lv is None:
            print("empty linkVector:%s" % (lk))
            continue
        cosine_list = []
        for ck, cv in content_vectors.items():
            if cv is None:
                print("empty contentVector:%s" % (ck))
                continue
            cos_val = cosine(lv, cv)
            cosine_list.append({"cid": ck, "score": cos_val})
        cosine_list.sort(key=lambda item: item["score"], reverse=True)
        cosine_list = cosine_list[:min(2, len(cosine_list))]
        link_cosines_dict[lk] = cosine_list

    for lck, lcvs in link_cosines_dict.items():
        ltitle = link_dict[lck]
        for index in range(0, len(lcvs)):
            lcosine = lcvs[index]
            print("%s\t%s\t%s\t%s\t%s\t%s" % (
                str(index), lck, lcosine["cid"], ltitle, content_dict[lcosine["cid"]], str(lcosine["score"])))


def get_value(val_dict, key):
    if key in val_dict:
        return val_dict[key]
    return None


def encode_text(text):
    if text is None:
        return None
    return text.encode('UTF8')


def parse_year(title):
    if title is None:
        return None
    new_title = re.sub(r'(www\.[a-z0-9\.\-]+)|([a-z0-9\.\-]+?\.com)', "", title)
    match_obj = re.match("(19[0-9]{2}|20[0-9]{2})", new_title)
    if match_obj:
        return match_obj.group(1)
    return None


def tfidf_knn(link_dict, idf_dict, output_path):
    splitor = b" "
    analyze_key = 'analyze'
    for lk, lv in link_dict.items():
        if analyze_key not in lv:
            continue
        if 'contents' not in lv:
            continue
        analyze_txt = encode_text(get_value(lv, analyze_key))
        if not analyze_txt:
            continue
        link_imdb = get_value(lv, 'imdb')
        link_year = get_value(lv, 'year') or parse_year(encode_text(lv['title']))
        link_count = count_dict(analyze_txt.split(splitor))
        contents = lv['contents']
        similar_list = []
        for cc in contents:
            if link_year is not None and link_year > 0:
                # 单集影视,年份必须一样
                epcount = get_value(cc, 'epcount')
                if epcount is not None and str(epcount).isdigit() and int(str(epcount)) <= 1:
                    content_year = get_value(cc, 'year')
                    if link_year != content_year:
                        continue
            if link_imdb is not None and link_imdb == get_value(cc, 'imdb'):
                cos_val = 0.9999
            else:
                sAnalyze = encode_text(cc['analyze'])
                analyze_count = count_dict(sAnalyze.split(splitor))
                cos_val = tfidf_cosine(link_count, analyze_count, idf_dict)
            if cos_val < 0.70:
                continue
            cc["score"] = cos_val
            similar_list.append(cc)
        similar_list.sort(key=lambda item: item["score"], reverse=True)
        similar_list = similar_list[:min(1, len(similar_list))]
        lv['contents'] = similar_list
    print("link_dict:" + str(len(link_dict.keys())))
    bulk_doc = {}
    bulk_doc["_index"] = "link"
    bulk_doc["_type"] = "table"
    with open(output_path, 'w') as outf:
        for _, link_json in link_dict.items():
            if 'contents' not in link_json:
                continue
            contents = link_json['contents']
            for index in range(0, len(contents)):
                content_json = contents[index]
                match_vo = {}
                match_vo['title'] = get_value(link_json, 'title')
                match_vo['season'] = get_value(link_json, 'season')
                match_vo['episode'] = get_value(link_json, 'episode')
                match_vo['target'] = get_value(content_json, 'id').split('_')[0]
                # match_vo['doctxt'] = get_value(content_json, 'analyze')
                match_vo['score'] = float("%.4f" % get_value(content_json, 'score'))
                # -1:失效,0:默认,1:有效,2:自动匹配,3:人工匹配
                match_vo['status'] = 2
                match_vo['utime'] = int(time.time())
                bulk_doc["_id"] = get_value(link_json, "id")
                outf.write(json.dumps({"update": bulk_doc}) + "\n")
                outf.write(json.dumps({"doc": match_vo}, ensure_ascii=False) + "\n")
                # print("%s\t%s\t%s\t%s\t%s\t%s" % (
                #     str(index), str(lck), str(lcosine["cid"]), str(link_analyze)
                #     , str("@" + content_analyze),
                #     str(lcosine["score"])))


def parse_args():
    """
    config args
    :return:
    """
    parser = argparse.ArgumentParser(description="word2vec match")
    parser.add_argument(
        '--emb_dict_path',
        type=str,
        required=True,
        help="The path of the emb dict to match")
    parser.add_argument(
        '--idf_dict_path',
        type=str,
        required=True,
        help="The path of the idf of movie")
    parser.add_argument(
        '--link_path',
        type=str,
        required=True,
        help="The path of the links")
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="match output path")
    return parser.parse_args()


def main():
    """
    select match content for link
    :return:
    """
    args = parse_args()

    link_dict = load_json_dict(args.link_path)
    idf_dict = load_pairs(args.idf_dict_path, b"\t")
    for k, v in idf_dict.items():
        idf_dict[k] = float(v)
    tfidf_knn(link_dict, idf_dict, args.output_path)
    # emb_dict = load_embedding(args.emb_dict_path)
    # w2v_knn(emb_dict, link_dict, content_dict, idf_dict)


if __name__ == '__main__':
    main()
    # ltoken_vector = [0.56264043, 0.6375902, 0.52622664, 0.]
    # rtoken_vector = [0.7303455, 0., 0.68307793, 0.]
    # lvector = np.array([float(x) for x in ltoken_vector])
    # rvector = np.array([float(x) for x in rtoken_vector])
    # cos_val = cosine(lvector, rvector)
    # print("cos_val:%s" % (str(cos_val)))
