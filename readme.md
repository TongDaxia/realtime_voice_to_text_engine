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
    python3.11 ./text_client/client.py --streaming --audio test_audio.m4a

客户端日志：
￼![图片](pic/client.png)

服务端日志：
￼![图片](pic/server.png)



# 常用命令

pip freeze > requirements.txt


linux环境部署办法

sudo vi /etc/systemd/system/realtime_py.service


sudo tee /etc/systemd/system/realtime_py.service <<'EOF'
[Unit]
Description=RealTime Text Speech Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/server/real_time_text/py
ExecStart=/bin/bash -c 'source /root/server/real_time_text/py/bin/activate && exec /root/server/real_time_text/py/run.sh start'
Restart=on-failure
RestartSec=5s
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/root/bin
Environment=PYTHONPATH=/root/server/real_time_text/py

[Install]
WantedBy=multi-user.target
EOF


sudo systemctl daemon-reload # 重新加载 Systemd 配置

sudo systemctl start realtime_py      # 立即启动
sudo systemctl enable realtime_py     # 开机自启

sudo systemctl status realtime_py #检查服务状态

sudo systemctl stop realtime_py       # 停止服务
sudo systemctl restart realtime_py    # 重启服务
sudo systemctl disable realtime_py    # 禁用开机自启
sudo systemctl status realtime_py    # 禁用开机自启

