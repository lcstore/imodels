# -*- coding: utf-8 -*-

"""

"""
import argparse
import json
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


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


def load_bulk_status_dict(token_path):
    """
    load bulk response to dict
    """
    bulk_status_dict = {}
    with open(token_path) as embf:
        for line in embf:
            line = line.strip()
            try:
                bulk_status_json = json.loads(line)
            except:
                continue
            if 'items' not in bulk_status_json:
                continue
            items = bulk_status_json['items']
            for item in items:
                for cmd, status_json in item.items():
                    if "_id" in status_json and "status" in status_json:
                        bulk_status_dict[status_json["_id"]] = status_json["status"]
    return bulk_status_dict


def parse_args():
    """
    config args
    :return:
    """
    parser = argparse.ArgumentParser(description="word2vec match")
    parser.add_argument(
        '--link_path',
        type=str,
        required=True,
        help="The path of the idf of movie")
    parser.add_argument(
        '--bulk_status_path',
        type=str,
        required=True,
        help="The path of bulk resp")
    parser.add_argument(
        '--remain_path',
        type=str,
        required=True,
        help="The path of bulk resp")

    return parser.parse_args()


def main():
    """
    remove link which bulk success
    :return:
    """
    args = parse_args()

    link_dict = load_json_dict(args.link_path)
    bulk_status_dict = load_bulk_status_dict(args.bulk_status_path)
    with open(args.remain_path, 'w') as outf:
        for lid, ljson in link_dict.items():
            if lid in bulk_status_dict and bulk_status_dict[lid] >= 200 and bulk_status_dict[lid] < 300:
                continue
            outf.write(json.dumps(ljson, ensure_ascii=False))


if __name__ == '__main__':
    main()
