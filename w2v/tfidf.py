# -*- coding: utf-8 -*-

"""

"""
import re
import math


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


def count_idf(doc_dict):
    token_count_dict = {}
    splitor = b" "
    total_doc_count = 0
    for k, v in doc_dict.items():
        total_doc_count = total_doc_count + 1
        unit_arr = v.split(splitor)
        for token in unit_arr:
            if token in token_count_dict:
                hash_doc_dict = token_count_dict[token]
            else:
                hash_doc_dict = {}
                token_count_dict[token] = hash_doc_dict
            hash_doc_dict[k] = 1
    idf_dict = {}
    for token, hash_doc_dict in token_count_dict.items():
        idf_dict[token] = math.log10(total_doc_count / len(hash_doc_dict.keys()))
    return idf_dict


def main():
    token_path = "testlocal/data/title.token"
    idf_path = "testlocal/data/content.idf"
    doc_dict = read_doc(token_path)
    token_count_dict = count_token(doc_dict)
    bad_token_dict = find_bad_token(token_count_dict)
    for k, v in bad_token_dict.items():
        print("bad:%s,count:%s" % (k, str(v)))
        # empty_bad_token(doc_dict, bad_token_dict)
    idf_dict = count_idf(doc_dict)
    idf_tuple_list = list(idf_dict.items())
    idf_tuple_list.sort(key=lambda item: item[1], reverse=True)
    for k, v in bad_token_dict.items():
        print("bad:%s,count:%s" % (k, str(v)))
        # empty_bad_token(doc_dict, bad_token_dict)
    with open(idf_path, 'w') as outf:
        for index in range(0, len(idf_tuple_list)):
            idf_tuple = idf_tuple_list[index]
            token = idf_tuple[0]
            idf = idf_tuple[1]
            if token in bad_token_dict:
                continue
            print("index:%s,token:%s,idf:%s" % (str(index), token, str(idf)))
            outf.write("%s\t%s\n" % (token, str(idf)))


if __name__ == '__main__':
    main()
