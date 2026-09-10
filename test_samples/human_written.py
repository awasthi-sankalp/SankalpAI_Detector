# parser.py  -  quick csv reader, written by a human

import csv, sys

def load(path):
    # open file, iterate rows
    rows = []
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        for row in rdr:
            rows.append(row)
    return rows

def dump(rows, sep='\t'):
    for r in rows:
        print(sep.join(r))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: parser.py <file>')
    dump(load(sys.argv[1]))
