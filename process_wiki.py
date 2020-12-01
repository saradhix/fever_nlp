import os
import json
import sys
WIKI_FOLDER = '/datasets/wiki-dump/wiki-pages/'
OUTPUT_FILE = '/datasets/wiki-dump/combined_replaced_pronouns'
def get_wiki_files(folderpath):
    files = os.listdir(folderpath)
    files = [os.path.join(folderpath,x) for x in files]
    return files

def main():
    ft=open(OUTPUT_FILE, 'w+')
    print("Hello")
    files = get_wiki_files(WIKI_FOLDER)
    print(files)
    for filename in files:
        print("Processing file", filename)
        fp=open(filename, 'r')
        for line in fp:
            obj = json.loads(line.strip())
            new_obj = {}
            #print(obj)
            page_id = obj['id']
            page_id_spaces = page_id.replace("-", " ")
            page_id_spaces = page_id.replace("_", " ")
            replace_dict = {'He': page_id_spaces, 'he': page_id_spaces, 'She':
                            page_id_spaces, 'she': page_id_spaces}
            new_obj[page_id]= []
            print("Page ID=", page_id)
            long_lines = obj['lines']
            #print("Long lines=", long_lines)
            for long_line in long_lines.split('\n'):
                fragments = long_line.split('\t')
                if len(fragments) <= 0:
                    print("ERROR: Less number of fragments for the line XXX", long_line ,"XXX")
                    continue
                line_number = fragments[0]
                #print("Fragments=", fragments)
                line = ' '.join(fragments[1:])
                #print("#", line_number, "line=", line)
                if len(line) <=1: continue
                words = line.split()
                words = [replace_dict.get(n, n) for n in words]
                line = ' '.join(words)
                new_obj[page_id].append((line_number, line))
            ft.write(json.dumps(new_obj)+'\n')
            #print(new_obj)
            #sys.exit()



        fp.close()
    ft.close()

if __name__ == "__main__":
    main()

