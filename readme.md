# 项目介绍

￼![图片](pic/实时语音转文字_时序图.png)




# 版本功能
server-3 client-1 基础识别

server-4 client-2 基础识别





# TODO List
1. 添加标点，为了提高速度，应该在识别结束的时候添加标点


# 测试方式

1. 启动python服务
    python speech_server3.py
2. 启动python client进行测试
    python ./text_client/client.py --streaming --audio test_audio.m4a

客户端日志：
￼![图片](pic/client.png)

服务端日志：
￼![图片](pic/server.png)



# 常用命令

pip freeze > requirements.txt
