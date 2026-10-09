#!/bin/bash
# 功能描述:

set -e
BASE_DIR=$(cd $(dirname $0);pwd)
# 时间版本
FLAG=$(date +%Y-%m-%d-%H-%M)

# 训练平台设定参数
echo "xiaoyiclaw_feature_task_develop_dataset_dir: ${DEVELOP_DATASET}"
echo "xiaoyiclaw_feature_task_product_dataset: ${PRODUCT_DATASET}"
echo "xiaoyiclaw_feature_task_output_dir: ${OUTPUT_DIR}"
echo "xiaoyiclaw_feature_task_log_dir: ${LOG_DIR}"
echo "xiaoyiclaw_feature_task_base_dir": ${BASE_DIR}
echo "xiaoyiclaw_feature_task_flag": ${FLAG}

parent_dir=$(dirname "$BASE_DIR")
# 自定义参数默认存储到算法目录下的train.config，检查 train.config 是否存在且不为空
if [ ! -f "${parent_dir}/train.config" ] || [ ! -s "${parent_dir}/train.config" ]; then
    echo "Error: train.config file is empty or does not exist!"
    exit 2
fi

cp "${parent_dir}"/train.config "${BASE_DIR}"/train.config

# 启动训练配置与特征提取分析任务
python ${BASE_DIR}/src/xiaoyiclaw_execute_feature_pipline.py \
    --DEVELOP_DATASET ${DEVELOP_DATASET} \
    --PRODUCT_DATASET ${PRODUCT_DATASET} \
    --OUTPUT_DIR ${OUTPUT_DIR} \
    --LOG_DIR ${LOG_DIR} \
    --BASE_DIR ${BASE_DIR} \
    --FLAG ${FLAG}
