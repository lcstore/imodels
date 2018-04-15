#!/usr/bin/env bash
python -u w2v/knearest.py \
--emb_dict_path=testlocal/data/emb_dict.vec \
--idf_dict_path=testlocal/data/content.idf \
--link_path=testlocal/data/link-title.json \
--output_path=testlocal/data/match-link.json