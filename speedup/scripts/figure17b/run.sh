FLEXGEN_PATH=$PWD/../../flexgen  # 设置 FLEXGEN_PATH 变量为当前工作目录的上级目录中的 flexgen 文件夹路径
rm $FLEXGEN_PATH/flexgen/flex_opt.py  # 删除 FLEXGEN_PATH 中的 flex_opt.py 文件，如果存在的话
rm $FLEXGEN_PATH/flexgen/pytorch_backend.py  # 删除 FLEXGEN_PATH 中的 pytorch_backend.py 文件，如果存在的话
ln -s ../infinigen/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py  # 创建一个指向 ../infinigen/flex_opt.py 的符号链接，命名为 flex_opt.py
ln -s ../infinigen/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py  # 创建一个指向 ../infinigen/pytorch_backend.py 的符号链接，命名为 pytorch_backend.py

for PARTIAL_WEIGHT_RATIO in 0.1 0.2 0.4 0.6 0.8 1.0  # 遍历部分权重比率的不同值
do
  CMD="--model huggingface/opt-13b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 8 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"  # 定义一个命令行参数字符串 CMD，其中包含模型参数和其他配置
  CMD=$CMD" --alpha 4 --partial-weight-ratio $PARTIAL_WEIGHT_RATIO --max-num-kv 409"  # 将 alpha、部分权重比率和最大 KV 数量的参数添加到 CMD 字符串中
  python -m flexgen.flex_opt $CMD  # 执行 flexgen.flex_opt 模块，并将 CMD 作为参数传递，开始模型的推理
done  # 结束 for 循环，继续下一个部分权重比率的迭代