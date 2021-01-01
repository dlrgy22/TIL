from elasticsearch import Elasticsearch

def insert_data(ip, data, index_name):
    es = Elasticsearch(ip)

    doc = {"x" : data[0],
           "y" : data[1]
            }

    res = es.index(index=index_name,
                   doc_type="log",
                   body=doc)
    print(doc)
    print(res)
