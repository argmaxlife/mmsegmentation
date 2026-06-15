#!/bin/bash
#
# GPU 空闲等待脚本 + 多训练任务队列
# 颜色输出 + 日志记录 + 模块化结构
#

########################################
# 💡 配置区
########################################

# 全局配置（只需修改这些部分）
GPUS="0,1,2,3,4,5,6,7"        # GPU 设置
NUM_GPUS=8                    # 每个任务使用的 GPU 数量
NUM_WORKERS=4                 # 数据加载的工作线程数
BATCH_SIZE=16                 # 每个 GPU 的 batch size
WARMUP=141                    # warmup
MAX_ITERS=42300               # 最大迭代次数
VAL_INTERVAL=1410             # 验证间隔
DATASET_NAME="sly_oilleak_20251010_v7.4.3"  # 数据集名称
DATA_ROOT="../../data/${DATASET_NAME}/mmseg" # 数据集路径
CONFIG_DIR="configs/custom"   # 配置文件目录
LOG_FILE="./gpu_wait_and_run.log" # 日志文件

# 需要训练的配置文件列表
CONFIG_FILES=(
    "oilleak-dinov3-adapterv3-vitb-convnexttiny.py"
)

INTERVAL=10  # 检测间隔秒数
MAX_GPU_UTIL=5          # 利用率 <=5% 视为空闲
MAX_MEM_USED_PERCENT=5  # 显存占比 <=5% 视为空闲

# 飞书机器人 Webhook 地址
WEBHOOK_URL="https://www.feishu.cn/flow/api/trigger-webhook/e0ab5dd358fa02c81664f9d6dea77f6a"

########################################
# 🎨 颜色函数
########################################
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

log_info()    { echo -e "${GREEN}[INFO]${RESET} $*" | tee -a "$LOG_FILE"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET} $*" | tee -a "$LOG_FILE"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*" | tee -a "$LOG_FILE"; }
log_section() { echo -e "${BLUE}===== $* =====${RESET}" | tee -a "$LOG_FILE"; }


########################################
# 🧪 检查依赖
########################################
if ! command -v nvidia-smi &>/dev/null; then
    log_error "未找到 nvidia-smi，请安装 NVIDIA 驱动"
    exit 1
fi

########################################
# 飞书消息发送函数
########################################
send_feishu_message() {
    local MESSAGE="$1"

    curl -s -X POST "$WEBHOOK_URL" \
         -H "Content-Type: application/json" \
         -d "{
              \"msg_type\": \"text\",
              \"content\": {
                  \"text\": \"$MESSAGE\"
              }
          }"

    # 检查是否发送成功（简单判断HTTP状态码）
    if [ $? -eq 0 ]; then
        log_info "消息发送成功！"
    else
        log_error "消息发送失败，请检查网络或Webhook地址。"
    fi
}


########################################
# 📌 判断 GPU 是否空闲
########################################
is_gpu_idle() {
    local idle=true

    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    if [ $? -ne 0 ]; then
        log_warn "nvidia-smi 查询失败，继续重试..."
        return 1
    fi

    INDEX=0
    while IFS=, read -r util used total; do
        mem_percent=$(echo "scale=2; $used / $total * 100" | bc)

        log_info "GPU $INDEX: util=${util}%  mem=${used}/${total} MiB (${mem_percent}%)"

        # 超过阈值则判定为忙
        if [ "$util" -gt "$MAX_GPU_UTIL" ]; then idle=false; fi
        if (( $(echo "$mem_percent > $MAX_MEM_USED_PERCENT" | bc -l) )); then idle=false; fi

        INDEX=$((INDEX + 1))
    done <<< "$GPU_INFO"

    $idle && return 0 || return 1
}


########################################
# 配置文件修改函数
########################################
update_config_file() {
    local cfg_file="$1"

    log_info "🔧 修改配置文件: $cfg_file"

    # 修改 batch_size, warmup, max_iters, val_interval
    sed -i "s/batch_size *= *[0-9]\+/batch_size=${BATCH_SIZE}/" "$cfg_file"
    sed -i "s/warmup *= *[0-9]\+/warmup = ${WARMUP}/" "$cfg_file"
    sed -i "s/max_iters *= *[0-9]\+/max_iters = ${MAX_ITERS}/" "$cfg_file"
    sed -i "s/val_interval *= *[0-9]\+/val_interval = ${VAL_INTERVAL}/" "$cfg_file"

    # 替换 data_root
    sed -i "s#data_root *= *[\"'][^\"']*[\"']#data_root = \"${DATA_ROOT}\"#" "$cfg_file"

    # 替换 num_workers
    sed -i "s/num_workers *= *[0-9]\+/num_workers=${NUM_WORKERS}/" "$cfg_file"
}


########################################
# 🚀 执行训练队列
########################################
run_training_tasks() {
    log_section "开始执行训练队列，总计 ${#CONFIG_FILES[@]} 条任务"

    for ((i=0; i<${#CONFIG_FILES[@]}; i++)); do
        CONFIG_FILE="${CONFIG_FILES[$i]}"
        CFG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"

        # 生成工作目录的后缀
        SUFFIX=$(echo "$CONFIG_FILE" | sed -E 's/oilleak(.*)\.py/\1/')
        WORK_DIR="../../experiments/${DATASET_NAME}${SUFFIX}"

        # 修改配置文件
        # update_config_file "$CFG_PATH"

        CMD="CUDA_VISIBLE_DEVICES=${GPUS} bash tools/dist_train.sh $CFG_PATH ${NUM_GPUS} --work-dir $WORK_DIR"
        log_section "执行任务 $((i+1))/$((${#CONFIG_FILES[@]})): $CMD"
        eval "$CMD"
        EXIT_CODE=$?

        log_info "任务 $((i+1)) 完成，退出码=$EXIT_CODE"
        echo "" | tee -a "$LOG_FILE"
    done

    log_section "所有训练任务执行完毕"
}


########################################
# 🕒 主循环：等待 GPU 空闲
########################################
log_section "脚本启动"
log_info "GPU 判空阈值: util ≤ $MAX_GPU_UTIL%, mem ≤ $MAX_MEM_USED_PERCENT%"
log_info "检测间隔: $INTERVAL 秒"
log_info "训练任务数: ${#CONFIG_FILES[@]}"

while true; do
    if is_gpu_idle; then
        log_section "GPU 空闲，开始执行训练任务"
        
        # 发送飞书消息
        send_feishu_message "GPU 空闲，开始执行训练任务"
        
        run_training_tasks
        exit 0
    else
        log_warn "GPU 忙碌中，等待 ${INTERVAL} 秒后继续检测..."
    fi

    sleep "$INTERVAL"
done
