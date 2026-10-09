#!/bin/bash
# 功能描述: 构建小艺claw任务prompt攻击检测模型包

set -e
BASE_DIR=$(cd $(dirname $0);pwd)

echo "product_dataset: ${PRODUCT_DATASET}"
echo "output_dir: ${OUTPUT_DIR}"
echo "flag": ${FLAG}
echo "version": ${VERSION}
echo "log_dir: ${LOG_DIR}"
echo "base_dir": ${BASE_DIR}

echo "------ start build pattern model package ------"
source_dir="${OUTPUT_DIR}/pattern/model/fuse_model/rf/${FLAG}"
cd "${OUTPUT_DIR}"

# 1.组装攻击模式模型包
mkdir -p ./model
cp "${BASE_DIR}/model_package_config/sysconfig.properties" ./model/

# 组装模型
pattern_model_dir="./model/AttackPatternDetectionModel_${VERSION}"

mkdir -p "${pattern_model_dir}"/config
parent_dir=$(dirname "$BASE_DIR")
cp "${parent_dir}"/train.config "${pattern_model_dir}"/config/attack_pattern_config.json

mkdir -p "${pattern_model_dir}"/knowledge_base
old_knowledge_base="${PRODUCT_DATASET}/pattern/knowledge_base/keyword_db/${FLAG}"
cp "${old_knowledge_base}"/bayes_prob.json "${pattern_model_dir}"/knowledge_base/bayes_prob.json
cp "${old_knowledge_base}"/docs.json "${pattern_model_dir}"/knowledge_base/docs.json
cp "${old_knowledge_base}"/keywords.json "${pattern_model_dir}"/knowledge_base/keywords.json
cp "${PRODUCT_DATASET}"/pattern/knowledge_base/stopwords.txt "${pattern_model_dir}"/knowledge_base/stopwords.txt
syntax_patterns_dir="${PRODUCT_DATASET}/pattern/knowledge_base/syntax_patterns/"
cp -r "${syntax_patterns_dir}" "${pattern_model_dir}"/knowledge_base/

mkdir -p "${pattern_model_dir}"/model
echo "${pattern_model_dir}"/model/fast_fusion_model.pkl
cp "${source_dir}"/model.pkl "${pattern_model_dir}"/model/fast_fusion_model.pkl
cp "${BASE_DIR}"/model_package_config/model_info.json "${pattern_model_dir}"/model/model_info.json

cd "model/"
cp -r "./" "${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}"/pattern
rm -rf "${OUTPUT_DIR}"/model

# 2.组装攻击意图模型
# 要处理几级模型，如1, 2, 3, 4
echo "model_level: ${MODEL_LEVEL}"

# 验证 MODEL_LEVEL 是否为 1-4 的整数
if ! [[ "${MODEL_LEVEL}" =~ ^[1-4]$ ]]; then
    echo "Error: the value of MODEL_LEVEL must be 1-4"
    exit 1
fi

echo "------ start build intention model package -----"
current_dir="${parent_dir}"/attack_intention_detection
target_dir="${current_dir}/package_output"
[ -d "${target_dir}" ] && rm -rf "${target_dir}"
[ ! -d "${target_dir}" ] && mkdir -p "${target_dir}"
cd "${target_dir}"

mkdir -p ./model

# 2.1 组装bert模型
bert_dir="./model/AttackIntentDetectionModel_bert_25.0.0/"
mkdir -p ${bert_dir}/config
cp "${current_dir}"/train_config/bert_config.json ${bert_dir}/config/bert_config.json
mkdir -p ${bert_dir}/model
cp "${PRODUCT_DATASET}"/model/L0/bert/*.json ${bert_dir}/model/
cp "${PRODUCT_DATASET}"/model/L0/bert/vocab.txt ${bert_dir}/model/
leve="L${MODEL_LEVEL}"
mkdir -p "${bert_dir}/model/convert_model"
cp "${PRODUCT_DATASET}/model/${leve}/bert/mindspore_model.ckpt" "${bert_dir}/model/convert_model"
cp "${PRODUCT_DATASET}/model/L2/centroids.npy" "${bert_dir}/model/convert_model"


# 2.2 组装分类模型
classify_dir="./model/AttackIntentDetectionModel_classify_25.0.0"
mkdir -p ${classify_dir}/config
# 使用 cat 写入 JSON，当作意图模型的train.config
cat << EOF > "${classify_dir}/config/classify_config.json"
{
  "feature_dim": ${feature_dim},
  "cluster_num": ${cluster_num},
  "content_length_threshold": ${content_length_threshold},
  "rf_threshold": ${rf_threshold}
}
EOF
cat ${classify_dir}/config/classify_config.json
mkdir -p ${classify_dir}/model
cp "${PRODUCT_DATASET}/model/L2/centroids.npy" ${classify_dir}/model/
cp "${current_dir}/train_config/classify_model_info.json" ${classify_dir}/model/model_info.json
cp "${PRODUCT_DATASET}/model/L4/clf/attack_intent_detection_classify_model.pkl" ${classify_dir}/model/


echo "start output package"
TIME_VERSION=$(date +%Y%m%d%H%M)
echo "time_version: ${TIME_VERSION}"
TEMP_DIR="/cache"
cd "model/"
cp -r "./" "${OUTPUT_DIR}"
echo "end output package"

echo "------ end build model package -----"
