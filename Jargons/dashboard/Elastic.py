import csv
import json
from elasticsearch import helpers, Elasticsearch

args = {
    'hosts': 'https://81d10e0823734aa493842a05391800b0.us-east-1.aws.found.io:9243/',
    'user': 'elastic',
    'secret': 'm68cHCrzHDgU64sdNgsfuP4J'
}
obj = {
    "_index": "",
    "_type": "",
    "_id": "",
    "_source": {}
}
es = Elasticsearch(args.get('hosts'), http_auth=(args.get('user'), args.get('secret')))


def csv_writer(file_name, index_name, doc_type, delimiter):
    tmp_file = 'tmp.csv'
    remove_quotes(file_name, tmp_file)
    headers = []
    count = 0

    obj['_index'] = index_name
    obj['_type'] = doc_type

    with open(tmp_file, 'r') as outfile:
        reader = csv.reader(outfile, delimiter=delimiter)

        actions = []
        for row in reader:
            if count == 0:
                headers = row
            else:
                tmp_obj = dict(obj)
                tmp_obj['_source'] = {}
                for i, val in enumerate(row):
                    tmp_obj['_source'][headers[i]] = '0' if val == 'null' else val
                tmp_obj['_id'] = count

                actions.append(dict(tmp_obj))
            count += 1
        # print(actions)
        helpers.bulk(es, actions=actions)
        print('done')


def csv_reader(index_name, size):
    doc = {
        'size': size
    }
    return es.search(index=index_name, body=doc, request_timeout=30)


def get_index_size(index_name, doc_type):
    return es.count(index=index_name, doc_type=doc_type)['count']


def get_indices():
    return es.indices.get_alias().keys()


def remove_quotes(old_file, new_file):
    with open(old_file, 'r') as input_file, open(new_file, 'w+') as output_file:
        file_data = input_file.read()
        output_file.write(file_data.replace('\"', ''))


# if __name__ == "__main__":
#     es.indices.delete('cpu_usage')
#     print(es.indices.get_alias().keys())
#     data = csv_reader(index_name='cpu_usage', size=10000)
#    print(json.dumps(data, indent=4))
#     df = Select.from_dict(data).to_pandas()
#     print(df)
#
#     csv_writer(file_name='CPU_usage.csv', index_name='cpu_usage', doc_type='data', delimiter=';')
#
