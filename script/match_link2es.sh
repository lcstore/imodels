#!/usr/bin/env bash
UNMATCH_DIR=$1
CUR_SCRIPT_DIR=$(cd `dirname $0`; pwd)
CUR_DATA_DIR=$(cd $CUR_SCRIPT_DIR; cd ../testlocal; pwd)
echo "CUR_DATA_DIR:$CUR_DATA_DIR,CUR_DIR:$CUR_SCRIPT_DIR"
OUT_JSON_LINK="$CUR_DATA_DIR/data/link-title.json"
OUT_BULK_MATCH="$CUR_DATA_DIR/data/match-link.json"
OUT_BULK_STATUS="$CUR_DATA_DIR/data/bulk-status.json"
OUT_UNMATCH_LINK="$CUR_DATA_DIR/data/unmatch-link-title.json"
CUR_DAY_DIR=`date -d"0 day" "+%Y%m%d%H%M"`
#CUR_DAY_DIR=`date +%Y%m%d%H%M` --mac
sh $CUR_SCRIPT_DIR/knearest.sh
curl -XPOST 'localhost:9200/_bulk' --data-binary "@$OUT_BULK_MATCH" > $OUT_BULK_STATUS
sh $CUR_SCRIPT_DIR/removeall.sh
mv $OUT_UNMATCH_LINK $UNMATCH_DIR/$CUR_DAY_DIR