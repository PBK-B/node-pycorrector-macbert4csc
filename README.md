<!--
 * @Author: Bin
 * @Date: 2023-11-23
 * @FilePath: /macbert/README.md
-->
# 中文纠错服务

> NodeJS 利用 `macbert4csc-base-chinese` 对中文段落进行纠错

## install

```
huggingface-cli download shibing624/macbert4csc-base-chinese --cache-dir ./models/.cache --local-dir ./models/shibing624/macbert4csc-bas
e-chinese
```

## start

```
node main.js 
```

## demo

GET <http://127.0.0.1:3000/checking?text={文本}>

```
{
  "original": "我脚的，少先队员因该为老人让坐，而不柿自己坐。为什么你会因为自已觉得累反而枪老认的座位呢？",
  "proofread": "我觉得，少先队员应该为老人让坐，而不该自己坐。为什么你会因为自己觉得累反而抢老人的座位呢？",
  "errors": []
}
```



## documents

<https://huggingface.co/docs/huggingface_hub/main/guides/cli>

<https://huggingface.co/shibing624/macbert4csc-base-chinese>

<https://huggingface.co/docs/transformers.js/index>


## citation

```
@software{pycorrector,
  author = {Xu Ming},
  title = {pycorrector: Text Error Correction Tool},
  year = {2021},
  url = {https://github.com/shibing624/pycorrector},
}
```

