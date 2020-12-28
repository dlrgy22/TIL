from elasticsearch import Elasticsearch

def insert_data(ip, data, index_name):
    es = Elasticsearch(ip)
    doc = {"time" : data[0],
           "data" : data[1]
            }

    res = es.index(index=index_name, doc_type="log", body=doc)
    print(doc)
    print(res)
