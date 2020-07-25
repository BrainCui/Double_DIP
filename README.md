## Double DIP复现

### 目录结构

```
├── /data/            测试数据
├── /log/             loss曲线的log
├── /net/             网络结构与损失函数
│   ├── Concat.py     连接结构
│   ├── DIP.py        DIP核心网络结构
│   └── losses.py     损失函数
├── draw.py           中间/最终结果展示函数
├── segmentation.py   图像前后景分割(主函数)
└── lab6_report.pdf   lab6 实验报告
```

### 运行代码

```bash
python segmentation.py
```



### 结果输出

中间及最终结果在根目录下``output``目录下（运行后会自动生成）