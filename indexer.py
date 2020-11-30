import re
from collections import defaultdict, OrderedDict
from os import path, listdir
import sys
import pickle
import time
import heapq
import json, nltk
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


#Need to change these below
INPUT_FILE = './wiki-pages/'
TITLES = './title_index/'
BODY = './body_index/'
# INDEX_STAT = '​invertedindex_stat.txt​'
TOTAL_TOKENS = 0
TOTAL_INV_TOKENS = 0
indexMap = defaultdict(list)
file_num = 0
pages = 1

def split_titles(txt):
    return txt.split('_')
def body_index(sentences, fstr, n):
    res = defaultdict(list)
    itr = 0
    for article in sentences:
        n_lines = article.split('\n')[:n]
        lst = []
        d = defaultdict(lambda: 0)
        for line in n_lines:
            temp = line.split()[1:]
            for j in temp:
                d[j] += 1
        for word in d.keys():
            string = fstr + '-' + str(itr) + '-' + str(d[word])
            res[word].append(string)
        itr += 1
    del sentences
    return res
def title_index(titles, fstr):

    res = defaultdict(list)
    itr = 0
    for lst in titles:
        lst = lst.split('_')
        d = defaultdict(lambda: 0)
        for j in lst:
            d[j] += 1
        for word in d.keys():
            string = fstr + '-' + str(itr) + '-' + str(d[word])
            res[word].append(string)
        itr += 1
    del titles
    return res

def store_index(ind, st, titles = True):
    index_map_file = []
    for key in sorted(ind.keys()):
        string = key + ':' + ' '.join(ind[key])
        index_map_file.append(string)
    if titles:
        with open(TITLES+st+'.txt' ,"w+") as f:
            f.write('\n'.join(index_map_file))
    else:
        with open(BODY+st+'.txt' ,"w+") as f:
            f.write('\n'.join(index_map_file))

def wiki_parse():
    print("Starting parser")
    file_num = 0
    total_time = 0
    for i in tqdm(range(1,110)):
        if i < 10:
            fstr = '00' + str(i)
        elif i < 100:
            fstr = '0' + str(i)
        else:
            fstr = str(i)
        train = []
        raw = open('wiki-pages/wiki-'+fstr+'.jsonl', 'r')
        for line in raw:
            obj = json.loads(line)
            train.append(obj)
        raw.close()
        df_wiki = pd.DataFrame(train)
        del train
        title_ind = title_index(df_wiki.id, fstr)
        store_index(title_ind, str(fstr))
        del title_ind
        body_ind = body_index(df_wiki.lines, fstr, 2)
        store_index(body_ind, str(fstr), titles=False)
        del body_ind
        del df_wiki
def merge_index_titles():
    global pages
    file_pointers = []
    for filename in listdir(TITLES):
        if filename.endswith('.txt'):
            print(TITLES+filename)
            file_pointers.append(open(TITLES+filename, 'r'))
    # ind = open('index/index','w+')
    # off = open('index/offset','w+')
    file_pointers_temp = file_pointers[0:40]
    last_file_count = 40
    file_pointers_flag = 1
    loop_count = 0
    while file_pointers_flag:
        print(last_file_count)
        wordpostings = defaultdict(lambda: [])
        words = {}
        heap = []
        wordfilemap = defaultdict(lambda: [])
        curline = {}
        finishflag = 1
        flag = 0
        filecomplete = [0 for i in range(len(file_pointers_temp))]
        ind = open(TITLES+'index_temp'+str(loop_count),'w+')
        off = open(TITLES+'offset_temp'+str(loop_count),'w+')
        for i in range(len(file_pointers_temp)):
            curline[i] = file_pointers_temp[i].readline().strip()
            # print(curline[i])
            word = curline[i].split(':')[0]
            wordfilemap[word].append(i)
            wordpostings[word] += curline[i].split(':')[1].split(" ")
            if word not in heap:
                heapq.heappush(heap,word)
        while (finishflag):
            minword = heapq.heappop(heap)
            string = minword + ':' + str(ind.tell())
            string = string.strip() + '\n'
            off.write(string)
            del string
            string = minword + ":" + " ".join(wordpostings[minword]) + "\n"
            ind.write(string)
            del string
            filenum = wordfilemap[minword]
            # print(wordfilemap)
            wordfilemap.pop(minword)
            wordpostings.pop(minword)
            for num in filenum:
                nextline = file_pointers_temp[num].readline().strip()
                if nextline == '':
                    filecomplete[num] = 1
                else:
                    newword = nextline.split(':')[0]
                    wordpostings[newword] += nextline.split(':')[1].split(" ")
                    # print(wordfilemap[newword])
                    if not wordfilemap[newword]:
                        heapq.heappush(heap,newword)
                        wordfilemap[newword].append(num)
                    else:
                        wordfilemap[newword].append(num)
            for i in range(len(file_pointers_temp)):
                flag = filecomplete[i] + flag
            flag = int(flag/(len(file_pointers_temp)))
            if flag==1:
                finishflag = 0
        for i in range(len(file_pointers_temp)):
            file_pointers_temp[i].close()
        ind.close()
        off.close()
        if file_pointers_flag==2:
            file_pointers_flag = 0
        elif last_file_count+39 >= len(file_pointers):
            file_pointers_temp = file_pointers[last_file_count:]
            file_pointers_flag = 2
        else:
            file_pointers_temp = file_pointers[last_file_count:last_file_count+39]
        file_pointers_temp.append(open(TITLES+'index_temp'+str(loop_count),'r'))
        last_file_count = last_file_count + 39
        loop_count += 1
        wordpostings = None
        words = None
        heap = None
        wordfilemap = None
        curline = None
    return

def merge_index_body():
    global pages
    file_pointers = []
    for filename in listdir(BODY):
        if filename.endswith('.txt'):
            print(BODY+filename)
            file_pointers.append(open(BODY+filename, 'r'))
    # ind = open('index/index','w+')
    # off = open('index/offset','w+')
    file_pointers_temp = file_pointers[0:40]
    last_file_count = 40
    file_pointers_flag = 1
    loop_count = 0
    while file_pointers_flag:
        print(last_file_count)
        wordpostings = defaultdict(lambda: [])
        words = {}
        heap = []
        wordfilemap = defaultdict(lambda: [])
        curline = {}
        finishflag = 1
        flag = 0
        filecomplete = [0 for i in range(len(file_pointers_temp))]
        ind = open(BODY+'index_temp'+str(loop_count),'w+')
        off = open(BODY+'offset_temp'+str(loop_count),'w+')
        for i in range(len(file_pointers_temp)):
            curline[i] = file_pointers_temp[i].readline().strip()
            # print(curline[i])
            word = curline[i].split(':')[0]
            wordfilemap[word].append(i)
            wordpostings[word] += curline[i].split(':')[1].split(" ")
            if word not in heap:
                heapq.heappush(heap,word)
        while (finishflag):
            minword = heapq.heappop(heap)
            string = minword + ':' + str(ind.tell())
            string = string.strip() + '\n'
            off.write(string)
            del string
            string = minword + ":" + " ".join(wordpostings[minword]) + "\n"
            ind.write(string)
            del string
            filenum = wordfilemap[minword]
            # print(wordfilemap)
            wordfilemap.pop(minword)
            wordpostings.pop(minword)
            for num in filenum:
                nextline = file_pointers_temp[num].readline().strip()
                if nextline == '':
                    filecomplete[num] = 1
                else:
                    newword = nextline.split(':')[0]
                    wordpostings[newword] += nextline.split(':')[1].split(" ")
                    # print(wordfilemap[newword])
                    if not wordfilemap[newword]:
                        heapq.heappush(heap,newword)
                        wordfilemap[newword].append(num)
                    else:
                        wordfilemap[newword].append(num)
            for i in range(len(file_pointers_temp)):
                flag = filecomplete[i] + flag
            flag = int(flag/(len(file_pointers_temp)))
            if flag==1:
                finishflag = 0
        for i in range(len(file_pointers_temp)):
            file_pointers_temp[i].close()
        ind.close()
        off.close()
        if file_pointers_flag==2:
            file_pointers_flag = 0
        elif last_file_count+39 >= len(file_pointers):
            file_pointers_temp = file_pointers[last_file_count:]
            file_pointers_flag = 2
        else:
            file_pointers_temp = file_pointers[last_file_count:last_file_count+39]
        file_pointers_temp.append(open(BODY+'index_temp'+str(loop_count),'r'))
        last_file_count = last_file_count + 39
        loop_count += 1
        wordpostings = None
        words = None
        heap = None
        wordfilemap = None
        curline = None
    return


def main():
    wiki_parse()
    merge_index_titles()
    merge_index_body()

if __name__ == "__main__":
    main()

