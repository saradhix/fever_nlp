import json
import sys

def main():
    page_id = sys.argv[1]
    fp=open('combined.jsonl', 'r')
    for line in fp:
        obj = json.loads(line.strip())
        if list(obj.keys())[0] == page_id:
            print(obj)

if __name__ == "__main__":
    main()
