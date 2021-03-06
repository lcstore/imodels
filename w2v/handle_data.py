# -*- coding: utf-8 -*-

"""

"""
import re


def count_token(doc_dict):
    splitor = b" "
    count_dict = {}
    for k, v in doc_dict.items():
        unitArr = v.split(splitor)
        for token in unitArr:
            if token in count_dict:
                count_dict[token] += 1
            else:
                count_dict[token] = 1
    return count_dict


def find_bad_token(count_dict):
    count_list = list(count_dict.items())
    count_list.sort(key=lambda item: item[1], reverse=True)
    bad_dict = {}
    for index in range(0, len(count_list)):
        count = count_list[index]
        token = count[0]
        num = count[1]
        # print("%s:%s" % (token, str(num)))
        if num > 1000 and len(token) < 2:
            bad_dict[token] = num
        else:
            match = re.match(r'(^www\.[a-z0-9\.\-]+)|([a-z0-9\.\-]+?\.com$)', token, re.I)
            if match:
                bad_dict[token] = num
    return bad_dict


def empty_bad_token(doc_dict, bad_dict):
    splitor = b" "
    for k, token in doc_dict.items():
        unitArr = token.split(splitor)
        for index in range(0, len(unitArr)):
            token = unitArr[index]
            if token in bad_dict:
                unitArr[index] = ""
        print("%s" % (splitor.join(unitArr)))


def read_doc(path):
    doc_dict = {}
    with open(path) as embf:
        for line in embf:
            unit_arr = line.strip().split(b"=")
            if len(unit_arr) < 2:
                continue
            id = unit_arr[0]
            token = unit_arr[1]
            doc_dict[id] = token
    return doc_dict


def main():
    token_path = "testlocal/data/emb_train.token"
    doc_dict = read_doc(token_path)
    token_count_dict = count_token(doc_dict)
    bad_token_dict = find_bad_token(token_count_dict)

    # bad_token_dict["imdb"] = 1
    # bad_token_dict["电影"] = 1
    # bad_token_dict["mp4"] = 1
    # bad_token_dict["rmvb"] = 1
    # bad_token_dict["双字"] = 1
    # bad_token_dict["ep0"] = 1
    # for k, v in bad_token_dict.items():
    #     print("bad:%s,count:%s" % (k, str(v)))
    empty_bad_token(doc_dict, bad_token_dict)


if __name__ == '__main__':
    main()
