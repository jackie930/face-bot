# face-bot

- 人脸比对机器人
功能：输入两张图片，返回是否为同一个人,返回值 true/false

- 人脸搜索机器人
功能：输入人脸图片,返回人名

## prepare
如果需要创建自己的人脸库，您需要手动启动endpoint

~~~ shell script
cd face_embedding_endpoint
python create_endpoint.py
~~~

等到sagemaker中生成对应的endpoint成功后，名为`insightface`


然后准备图片，将图片命名为x.png (x为姓名）放在`face_search_bot/dependency/pics`目录下，生成搜索空间，并更新您自己的ecr image
~~~

pip install opencv-python==3.4.11.41 -i https://opentuna.cn/pypi/web/simple
pip install annoy==1.15.2 -i https://opentuna.cn/pypi/web/simple
#注意修改create_search_space_ann.py中region名
python create_search_space_ann.py 
~~~

您会看到生成了`name.txt`,`test.ann`，然后运行`bash build_and_push.sh`

## quick start on bot
使用byob注册机器人
~~~json
{
   "bot_name":"face_search_bot",
   "file_types":[
      "[".png",".jpg",".jpeg"]"
   ],
   "bot_image":"847380964353.dkr.ecr.us-west-2.amazonaws.com/face-search-bot",
   "bot_image_cmd":"",
   "endpoint_name":"insightface",
   "endpoint_ecr_image_path":"847380964353.dkr.ecr.us-west-2.amazonaws.com/insightface-sagemaker-inference",
   "instance_type":"ml.g4dn.xlarge",
   "bot_vcpu":"4",
   "bot_mem":"4096",
   "model_s3_path":"",
   "create_date":"2020-07-27 21:39:00",
   "update_date":"2020-07-27 22:39:00"
}
~~~

使用机器人
```json
{
    "s3_bucket":"bot-ocr-test-bucket", 
    "s3_path":"face/",
    "bot_name":"face_search_bot", 
    "number_of_bots":"1",
    "bulk_size":"500", 
    "output_s3_bucket":"bot-ocr-test-bucket",
    "output_s3_prefix":"face/output"
}
```

