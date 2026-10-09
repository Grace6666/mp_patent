#!/bin/bash
# 功能描述: 构建模型包

set -e
BASE_DIR=$(cd $(dirname $0);pwd)

echo "develop_dataset_dir: ${DEVELOP_DATASET}"
echo "product_dataset: ${PRODUCT_DATASET}"
echo "output_dir: ${OUTPUT_DIR}"
echo "log_dir: ${LOG_DIR}"
echo "base_dir": ${BASE_DIR}
echo "flag": ${FLAG}

echo "------ start build model package ------"
source_dir="${OUTPUT_DIR}/model/fuse_model/rf/${FLAG}"
target_dir="${OUTPUT_DIR}/modelDir"
[ -d "${target_dir}" ] && rm -rf "${target_dir}"
[ ! -d "${target_dir}" ] && mkdir -p "${target_dir}"
cd "${target_dir}"

# 组装模型包
mkdir -p ./data
mkdir -p ./meta
mkdir -p ./model

cp "${BASE_DIR}/model_package_config/type.mf" ./meta/
cp "${BASE_DIR}/model_package_config/sysconfig.properties" ./model/

# 组装模型
pattern_model_dir="./model/AttackPatternDetectionModel_25.0.0"

mkdir -p "${pattern_model_dir}"/config
cp "${BASE_DIR}"/train.config "${pattern_model_dir}"/config/attack_pattern_config.json

mkdir -p "${pattern_model_dir}"/knowledge_base
old_knowledge_base="${PRODUCT_DATASET}/fast_detector/keyword_db/${FLAG}"
cp "${old_knowledge_base}"/bayes_prob.json "${pattern_model_dir}"/knowledge_base/bayes_prob.json
cp "${old_knowledge_base}"/docs.json "${pattern_model_dir}"/knowledge_base/docs.json
cp "${old_knowledge_base}"/keywords.json "${pattern_model_dir}"/knowledge_base/keywords.json
cp "${PRODUCT_DATASET}"/utils/cn_stopwords.txt "${pattern_model_dir}"/knowledge_base/stopwords.txt

mkdir -p "${pattern_model_dir}"/model
cp "${source_dir}"/model.pkl "${pattern_model_dir}"/model/fast_fusion_model.pkl
cp "${BASE_DIR}"/model_package_config/model_info.json "${pattern_model_dir}"/model/model_info.json

echo "start zip package"
cd ..
TEMP_DIR="/cache"
model_package_name="/attack_pattern_detection_model_${FLAG}.zip"
zip -r "${TEMP_DIR}/${model_package_name}" ./
cp "${TEMP_DIR}/${model_package_name}" "${OUTPUT_DIR}/"

cp "${source_dir}/result.txt" ./result.txt
rm -rf "${OUTPUT_DIR}"/model
rm -rf "${OUTPUT_DIR}"/modelDir
echo "end zip package"

echo "------ end build model package ------"
