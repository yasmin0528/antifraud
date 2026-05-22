# antifraud
环境配置
安装：

```bash
pip install -r requirements_llm.txt
```bash
pip install -r requirements_antifraud.txt
```

建议：

LLM 与 AntiFraud 使用两个 conda 环境：

```bash
conda create -n llm python=3.10 -y
conda create -n antifraud python=3.10 -y
```

避免 vLLM、xformers、PyTorch Geometric 依赖冲突。

训练

 # 默认训练（MPFC with LLM）
 python train.py --config configs/default.yaml --name mpfc_llm \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --epochs 50

 测试

 # 加载最佳 checkpoint 做测试
 python train.py --config configs/default.yaml --name mpfc_llm_test \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --test --checkpoint outputs/mpfc_llm/ckpt/model_best.pt

 消融实验（一次性跑完所有）

 # 包含 wo_llm（保留MPFC但去掉LLM）+ wo_ca1/wo_ca3/wo_mpfc/wo_vta
 python train.py --config configs/default.yaml --name ablation_all \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --ablation --epochs 50

 单独跑 wo_llm 消融：

 # 只跑 wo_llm（验证 LLM 作为 mPFC 内置组件的贡献度）
 python train.py --config configs/ablation/wo_llm.yaml --name ablation_wo_llm \
     --epochs 50

 参数敏感性实验

 # batch size 扫描
 python train.py --config configs/sensitivity/batch_size.yaml --name sweep_batch_size \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --sweep --epochs 50

 # learning rate 扫描
 python train.py --config configs/sensitivity/learning_rate.yaml --name sweep_lr \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --sweep --epochs 50

 # hidden_dim 扫描
 python train.py --config configs/sensitivity/hidden_dim.yaml --name sweep_hidden \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --sweep --epochs 50

 # dropout 扫描
 python train.py --config configs/sensitivity/dropout.yaml --name sweep_dropout \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --sweep --epochs 50

 # pos_weight 扫描
 python train.py --config configs/sensitivity/pos_weight.yaml --name sweep_pos_weight \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --sweep --epochs 50

 多种子重复实验

 python train.py --config configs/default.yaml --name multi_seed \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --multi_seed --epochs 50

 推理输出到 README

 # 先用 test 模式跑出结果，然后我把结果写到 README
 python train.py --config configs/default.yaml --name mpfc_inference \
     --llm_api_url http://172.21.80.1:11434/v1/chat/completions \
     --test --checkpoint outputs/mpfc_llm/ckpt/model_best.pt 2>&1 | tee outputs/mpfc_inference/test_result.txt