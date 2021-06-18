# -*- coding: utf-8 -*-
# @Time    : 10/12/20 4:54 PM
# @Author  : Jackie
# @File    : create_search_space_ann.py
# @Software: PyCharm
from boto3.session import Session
import cv2
import json
import sys
import os

from annoy import AnnoyIndex


region_name = 'cn-northwest-1'  # os.getenv("region_name")
endpoint_name = 'insightface'  # os.getenv("endpoint_name")


# Hack to print to stderr so it appears in CloudWatch.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


eprint(">>>> start the job with param - region_name-{}, endpoint_name-{}, ".format(region_name, endpoint_name))

def create_search_space_main(
        input_path_folder,
        endpoint_name,
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
    print("start!")

    #get image list
    input_path_list = os.listdir(input_path_folder)
    labeled_data = {}
    i = 0
    name_list = []
    for path in input_path_list:
        # read image
        try:
            path = os.path.join(input_path_folder,path)
            print("process:", path)
            img = cv2.imread(path)

            # infer endpoint
            result = invoke_endpoint(session, endpoint_name, img)
            res = json.loads(result)
            ls = res['Output']['Feature'][1:-1].split(',')
            v = [float(i) for i in ls]
            labeled_data[i] = v
            i = i+1

            #append name
            #file name need to be folder_path/xx_name.type
            name = path.split('/')[-1].split('.')[0].split('_')[-1]
            name_list.append(name)
            print ("<<< process end for image", name)
        except:
            continue

    file= open('./name.txt', 'w')
    for fp in name_list:
        file.write(str(fp))
        file.write('\n')
    file.close()

    print ('create search ann file')

    f = 512
    t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
    for key, value in labeled_data.items():
        print(key, value)
        t.add_item(key, value)

    t.build(100) # 10 trees
    t.save('test.ann')

    print ("Done save file!")
    return labeled_data

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
    input_path_folder='./pics'
    create_search_space_main(input_path_folder, endpoint_name, region_name)
    eprint("<<< Exit.")