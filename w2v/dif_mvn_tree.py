# -*- coding: utf-8 -*-

"""

"""
import re
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def mvn_lines(path):
    line_arr = []
    with open(path) as embf:
        for line in embf:
            line = line.strip()
            match_obj = re.match(r'[^\.]*?([a-zA-Z0-9\\-]{2,}[\.:].*)', line, re.IGNORECASE)
            if match_obj:
                # print("line:%s,   @match_obj:%s" % (line, match_obj.group(1)))
                line_arr.append(match_obj.group(1))
    return line_arr


def main():
    """
    remove link which bulk success
    :return:
    """
    left_path = "/apps/src/codes/cpu-uap/uap-offline/tr"
    right_path = "/apps/src/codes/baidu/cpu/cpu-uap/uap-offline/tr"
    llines = mvn_lines(right_path)
    rlines = mvn_lines(left_path)
    rContent = "\n".join(rlines)
    for line in llines:
        rContent = rContent.replace(line, "")
    print("rContent:" + rContent)


if __name__ == '__main__':
    main()
