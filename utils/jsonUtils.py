import json


def save2json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def json2dict(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d


def pprint(data):
    data = json.dumps(data, indent=4, ensure_ascii=False,
                      sort_keys=False, separators=(',', ': '))
    print(data)


if __name__ == '__main__':
    filename = './files/test.json'
    savepath = './files/test_1.json'
    jsondata = json2dict(filename)
    pprint(jsondata)
    save2json(jsondata, savepath)    
