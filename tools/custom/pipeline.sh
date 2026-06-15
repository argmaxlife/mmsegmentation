#!/bin/bash

###############################
# 全局变量区（只需改这里）
###############################

# GPU 设置
GPUS="1,2,3,4,6,7"
NUM_GPUS=6

# 数据集名称（配置文件中 data_root 会引用它）
DATASET_NAME="sly_oilleak_20251010_v7.4.3"

# data_root 的最终路径（构建完整路径）
DATA_ROOT="../../data/${DATASET_NAME}/mmseg"

# config 文件目录
CONFIG_DIR="configs/custom"

# 需要训练的配置文件列表
CONFIG_FILES=(
    "oilleak-b-384.py"
    "oilleak-s-224.py"
    "oilleak-t-224.py"
    "oilleak-b-384-512x512.py"
    "oilleak-s-224-512x512.py"
    "oilleak-t-224-512x512.py"
)

# 超参
BATCH_SIZE=16
WARMUP=188
MAX_ITERS=56400
VAL_INTERVAL=1880
NUM_WORKERS=4

###############################
# 配置文件修改函数
###############################
update_config_file() {
    local cfg_file="$1"

    echo "🔧 修改配置文件: $cfg_file"

    # 修改 batch_size
    sed -i "s/batch_size *= *[0-9]\+/batch_size=${BATCH_SIZE}/" "$cfg_file"

    # 修改 warmup / max_iters / val_interval
    sed -i "s/warmup *= *[0-9]\+/warmup = ${WARMUP}/" "$cfg_file"
    sed -i "s/max_iters *= *[0-9]\+/max_iters = ${MAX_ITERS}/" "$cfg_file"
    sed -i "s/val_interval *= *[0-9]\+/val_interval = ${VAL_INTERVAL}/" "$cfg_file"

    # 完整替换 data_root 行
    sed -i "s#data_root *= *[\"'][^\"']*[\"']#data_root = \"${DATA_ROOT}\"#" "$cfg_file"

    # 替换 num_workers
    sed -i "s/num_workers *= *[0-9]\+/num_workers=${NUM_WORKERS}/" "$cfg_file"
}

###############################
# 主循环
###############################
for cfg in "${CONFIG_FILES[@]}"; do
    CFG_PATH="${CONFIG_DIR}/${cfg}"

    # 根据配置文件名解析 work-dir 后缀，例如：oilleak-b-384.py → -b-384
    SUFFIX=$(echo "$cfg" | sed -E 's/oilleak(.*)\.py/\1/')
    WORK_DIR="../../experiments/${DATASET_NAME}${SUFFIX}"

    echo "============================================="
    echo "▶ 训练配置:  $CFG_PATH"
    echo "▶ data_root: ${DATA_ROOT}"
    echo "▶ Work-dir:  ${WORK_DIR}"
    echo "============================================="

    # 修改配置文件
    update_config_file "$CFG_PATH"

    # 执行训练
    CUDA_VISIBLE_DEVICES=$GPUS bash tools/dist_train.sh "$CFG_PATH" $NUM_GPUS --work-dir "$WORK_DIR"
done
