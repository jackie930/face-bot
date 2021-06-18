#!/usr/bin/env python
# encoding: utf-8

import os
import sys
sys.path.append("./")

import json
from boto3.session import Session
import boto3
from pprint import pprint
import cv2
import re
import pandas as pd

from elasticsearch import Elasticsearch, RequestsHttpConnection
from datetime import datetime
import compare_face

region_name = os.getenv("region_name")
endpoint_name = os.getenv("endpoint_name")
output_s3_bucket = os.getenv("output_s3_bucket")
output_s3_prefix = os.getenv("output_s3_prefix")
elastic_search_host = os.getenv("es_host")
elastic_search_port = os.getenv("es_port")
elastic_search_protocol = os.getenv("es_protocol")
batch_id = os.getenv("batch_id")
job_id = os.getenv("job_id")
elastic_search_index = os.getenv("es_index")


# Hack to print to stderr so it appears in CloudWatch.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


eprint(
    ">>>> start the job with param - outside es_host-{}, batch_id - {}, region_name-{}, endpoint_name-{},\n \
     output_s3_bucket-{}, output_s3_prefix-{}, es_port-{}, es_protocol-{}, job_id-{}, ".format(
        elastic_search_host, batch_id, region_name, endpoint_name, output_s3_bucket, output_s3_prefix, elastic_search_port, elastic_search_protocol, job_id))

def face_compare_main(
    input_s3_path_list,
    endpoint_name,
    output_s3_bucket,
    output_s3_prefix,
    region_name
):
    """
 function: save json_files back to s3 (key: file name value: tag label)
    :param input_s3_path_list: 读入s3文件路径list
    :param endpoint_path: 保存名称
    """
    session = Session(
        region_name = region_name
    )
    es = __connect_ES()
    s3 = session.client("s3")
    print("start!")
    pprint(input_s3_path_list[:5])

    folder_list = [re.split('[\/\//]', i)[-2] for i in input_s3_list]
    df = pd.DataFrame({'folder':folder_list,'path':input_s3_list})

    # loop over sub model list
    for i in list(set(folder_list)):
        print("<<<<< deal for folder ",i)
        path_list = df[df['folder']==i].path.unique()
        print (path_list)
        file_list = []
        #file_list should contain 2 images
        for i in path_list:
            s3_path = "s3://{}/{}".format(x["_source"]["bucket"], x["_source"]["file_key"])
            result = {}
            bucket = s3_path.split("/")[2]
            key = "/".join(s3_path.split("/")[3:])
            file_name = s3_path.split("/")[-1]
            print(bucket, key, file_name)
            s3.download_file(Filename=file_name, Key=key, Bucket=bucket)
            file_list.append(file_name)

        # infer endpoint
        label = compare_face(region_name.endpoint_name, file_list)

        result[str(path_list)] = label
        # save json file
        json_file = file_name.replace(".jpg", ".json")
        with open(json_file, "w") as fw:  # 建议改为.split('.')
            json.dump(result, fw, ensure_ascii=False)
            print("write json file success!")

        # output to s3
        upload_key=output_s3_prefix+'/'+json_file
        s3.upload_file(Filename=json_file, Key=upload_key, Bucket=output_s3_bucket)
        print("uploaded to s3://{}/{}".format(output_s3_bucket, upload_key))

        # update elasticsearch
        doc_id = x["_id"]
        update_status_by_id(es, doc_id, status="COMPLETED", output=label)

        # delete file locally
        delete_file(json_file)
        delete_file(file_name)


def delete_file(file):
    """
    delete file
    :param file:
    :return:
    """
    if os.path.isfile(file):
        try:
            os.remove(file)
        except:
            pass


def __connect_ES() -> Elasticsearch:
    eprint(
        "Connecting to the ES Endpoint {}:{}".format(
            elastic_search_host, elastic_search_port
        )
    )
    es = None
    try:
        if "https" == elastic_search_protocol:
            print("Using https.")
            es = Elasticsearch(
                hosts=[{'host': elastic_search_host, 'port': elastic_search_port}],
                use_ssl=True
            )
        else:
            print("Using http.")
            es = Elasticsearch(
                hosts=[{"host": elastic_search_host, "port": elastic_search_port}],
                connection_class=RequestsHttpConnection,
            )

    except Exception as E:
        eprint("Unable to connect to {0}")
        eprint(E)
        exit(3)

    return es


def __search_for_file_list(job_id: str, batch_id: str, status="NOT_STARTED") -> list:
    """
    Fot bot to get their file to process.
    :param job_id: One customer request for bots to process on certain folder is one *job*, and thus have a job_id.
    :param batch_id: Files in one job will spited based on the number of bots. Bot id and batch id is the same.
    :param status: one of "NOT_STARTED, COMPLETED, PROCESSING"
    :return:
    """
    es = __connect_ES()
    query = {  # TODO Would speed up by reduce the result fields
        "size": 10000,
        "query": {
            "bool": {
                "must": [
                    {"match": {"job_id": job_id}},
                    {"match": {"batch_id": batch_id}},
                    {"match": {"status": status}},
                ]
            }
        },
    }
    eprint(">>> going to query: {}".format(query))

    resp = es.search(index=elastic_search_index, body=query, scroll="9s")

    # keep track of pass scroll _id
    old_scroll_id = resp["_scroll_id"]
    all_hits = resp["hits"]["hits"]
    # use a 'while' iterator to loop over document 'hits'
    while len(resp["hits"]["hits"]):

        # make a request using the Scroll API
        resp = es.scroll(
            scroll_id=old_scroll_id,
            scroll="2s",  # length of time to keep search context
        )

        # check if there's a new scroll ID
        if old_scroll_id != resp["_scroll_id"]:
            eprint("NEW SCROLL ID:", resp["_scroll_id"])

        # keep track of pass scroll _id
        old_scroll_id = resp["_scroll_id"]

        # eprint the response results
        eprint("\nresponse for index: " + elastic_search_index)
        eprint("_scroll_id:", resp["_scroll_id"])
        eprint('response["hits"]["total"]["value"]:{}'.format(resp["hits"]["total"]))

        # iterate over the document hits for each 'scroll'
        all_hits.extend(resp["hits"]["hits"])
    eprint("DOC COUNT:", len(all_hits))

    return all_hits


def update_status_by_id(es, doc_id, status="COMPLETED", output=""):
    resp = es.update(
        index=elastic_search_index,
        id=doc_id,
        body={
            "doc": {
                "status": status,
                "output": output,
                "complete_date": datetime.utcnow(),
            }
        },
        doc_type="_doc",
    )
    return resp


if __name__ == "__main__":
    eprint(">>> Start execution.")

    file_list = __search_for_file_list(job_id=job_id, batch_id=batch_id)
    face_compare_main(file_list, endpoint_name, output_s3_bucket, output_s3_prefix, region_name)
    eprint("<<< Exit.")
