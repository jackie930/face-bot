# -*- coding: utf-8 -*-
# @Time    : 10/12/20 4:54 PM
# @Author  : Jackie
# @File    : create_search_space_ann.py
# @Software: PyCharm
from boto3.session import Session
import cv2
import json
import numpy as np

import sys


region_name = 'cn-northwest-1'  # os.getenv("region_name")
endpoint_name = 'insightface'  # os.getenv("endpoint_name")


# Hack to print to stderr so it appears in CloudWatch.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


eprint(">>>> start the job with param - region_name-{}, endpoint_name-{}, ".format(region_name, endpoint_name))

def main(
        region_name,
        endpoint_name,
        file_list
):
    """
 function: save json_files back to s3 (key: file name value: tag label)
    :param input_s3_path_list: 读入s3文件路径list
    :param endpoint_path: 保存名称
    """
    session = Session(
        region_name = region_name
    )
    print("start!")

    v_list = []
    for file_path in file_list:
        print("process:", file_path)
        img = cv2.imread(file_path)

        # infer endpoint
        eprint ('<<<< start endpoint execution')
        result = invoke_endpoint(session, endpoint_name, img)
        eprint ('<<<< end endpoint execution')
        res = json.loads(result)
        ls = res['Output']['Feature'][1:-1].split(',')
        v = [float(i) for i in ls]
        v_list.append(v)

    #do compare
    sim = np.dot(np.array(v_list[0]), np.array(v_list[1]).T)
    print(sim)

    if sim>0.6:
        label = True
    else:
        label = False

    eprint ("<<<< label : ", label)
    return label

def preprocess(image):
    """
 function: preprocess imageg into json string
    """
    #     ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    #     img = mx.nd.array(image, ctx)

    #     # transform
    #     input_size = 224
    #     crop_ratio = 0.875
    #     resize = int(math.ceil(input_size / crop_ratio))
    #     transform_size = transforms.Compose(
    #         [
    #             transforms.Resize(resize),
    #             transforms.CenterCrop(input_size),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #         ]
    #     )
    #     img = transform_size(img)
    #     # print(img.shape)
    #     img_tensor = img.expand_dims(0).asnumpy()
    #     print(img_tensor.shape)
    #     img_tensor = np.reshape(img_tensor, (img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3]))
    #     print(img_tensor.shape)

    #     img_str = str(img_tensor.tolist())
    #     img_dict_str = '{"data": %s}' % img_str

    image = cv2.resize(image, (224, 224))
    #     print(image.shape)
    img_dict_str = '{"data": %s}' % str(image.tolist())
    #     print('img_dict_str:', img_dict_str)
    #     import pickle
    #     pickle.dump(img_dict_str, open('img_dict_str.pkl', 'wb'))

    return img_dict_str


def invoke_endpoint(session, endpoint_name, img):
    """
 function: use endpoint to infer on one single text
    """
    # first preprocess input text
    # print(endpoint_name)
    data = preprocess(img)
    runtime = session.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=data,  # json.dumps(data),
    )

    outputs = response["Body"].read()
    outputs = outputs.decode("utf-8")
    print(outputs)

    # return str(outputs)
    return outputs


if __name__ == "__main__":
    eprint(">>> Start execution.")
    file_list = ['test.jpg','test2.jpg']
    main(region_name, endpoint_name, file_list)
    eprint("<<< Exit.")