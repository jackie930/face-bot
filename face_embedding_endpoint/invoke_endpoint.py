#!/usr/bin/env python
# encoding: utf-8


from boto3.session import Session

import cv2
import sys


region_name = 'cn-northwest-1'  # os.getenv("region_name")
endpoint_name = 'insightface'  # os.getenv("endpoint_name")


# Hack to print to stderr so it appears in CloudWatch.
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


eprint(">>>> start the job with param - region_name-{}, endpoint_name-{}, ".format(region_name, endpoint_name))

def face_embedding_main(
    input_path_list,
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
    
    for path in input_path_list:
    
        # read image
        print("process:", path)
        img = cv2.imread(path)

        # infer endpoint
        result = invoke_endpoint(session, endpoint_name, img)
        
        print('result:', result)


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

    file_list = ['./test.png']
    face_embedding_main(file_list, endpoint_name, region_name)
    eprint("<<< Exit.")
