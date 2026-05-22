#!/usr/bin/env bash
# ============================================================
# run_all_experiments.sh
# AML 反洗钱模型 —— 全面实验脚本
#
# 功能：
#   1. 基础实验（baseline）
#   2. 消融实验（ablation: wo_ca1, wo_ca3, wo_mpfc, wo_vta, wo_llm）
#   3. 参数敏感性实验（sensitivity: batch_size, lr, hidden_dim, dropout, pos_weight）
#   4. 多随机种子实验（multi-seed: seeds=[42,123,456,789,1111]）
#   5. 各实验汇总结果展示
#
# 用法：
#   bash run_all_experiments.sh                    # 运行全部实验
#   bash run_all_experiments.sh --llm_api_url <URL> # 指定 LLM API
#   bash run_all_experiments.sh --only baseline     # 仅运行 baseline
#   bash run_all_experiments.sh --skip multi_seed   # 跳过多种子实验
#   bash run_all_experiments.sh --dry-run           # 仅打印命令不执行
# ============================================================

set -euo pipefail

# -------- 默认参数 --------
LLM_API_URL="http://localhost:11434/v1/chat/completions"
OUTPUT_DIR="outputs"
DRY_RUN=false
RUN_ALL=true
RUN_BASELINE=false
RUN_ABLATION=false
RUN_SENSITIVITY=false
RUN_MULTI_SEED=false
CONFIG_DIR="configs"
DATA_PATH="/root/autodl-tmp/dataset.csv"
PREPROCESSED_PATH="preprocessed_data.pt"

# -------- 解析命令行参数 --------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm_api_url)
            LLM_API_URL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --preprocessed_path)
            PREPROCESSED_PATH="$2"
            shift 2
            ;;
        --only)
            RUN_ALL=false
            case "$2" in
                baseline)   RUN_BASELINE=true ;;
                ablation)   RUN_ABLATION=true ;;
                sensitivity) RUN_SENSITIVITY=true ;;
                multi_seed) RUN_MULTI_SEED=true ;;
                *)
                    echo "Unknown experiment type: $2"
                    echo "Valid: baseline, ablation, sensitivity, multi_seed"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --skip)
            case "$2" in
                baseline)   RUN_BASELINE=false ;;
                ablation)   RUN_ABLATION=false ;;
                sensitivity) RUN_SENSITIVITY=false ;;
                multi_seed) RUN_MULTI_SEED=false ;;
                *)
                    echo "Unknown experiment type: $2"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash run_all_experiments.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --llm_api_url <URL>      LLM API endpoint (default: $LLM_API_URL)"
            echo "  --output_dir <DIR>        Output directory (default: $OUTPUT_DIR)"
            echo "  --data_path <PATH>        Raw data CSV path"
            echo "  --preprocessed_path <PATH> Preprocessed data path"
            echo "  --only <TYPE>             Run only one experiment type"
            echo "                            (baseline|ablation|sensitivity|multi_seed)"
            echo "  --skip <TYPE>             Skip one experiment type"
            echo "  --dry-run                 Print commands without executing"
            echo "  --help, -h                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

if $RUN_ALL && ! $RUN_BASELINE && ! $RUN_ABLATION && ! $RUN_SENSITIVITY && ! $RUN_MULTI_SEED; then
    RUN_BASELINE=true
    RUN_ABLATION=true
    RUN_SENSITIVITY=true
    RUN_MULTI_SEED=true
fi

# -------- 辅助函数 --------
PYTHON="python"

run_cmd() {
    local description="$1"
    shift
    echo ""
    echo "======================================================================"
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] $description"
    echo "  Command: $*"
    echo "======================================================================"
    if $DRY_RUN; then
        echo "  (DRY RUN - skipped)"
    else
        $PYTHON "$@"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "  [ERROR] Command failed with exit code $exit_code"
            echo "  Command: $*"
            # 继续执行，不中断整个脚本
        fi
    fi
}

# ============================================================
# 1. 基础实验 (Baseline)
# ============================================================
run_baseline() {
    echo ""
    echo "████████████████████████████████████████████████████████████████████"
    echo "  基础实验 (Baseline) - 完整 MPFC 模型"
    echo "████████████████████████████████████████████████████████████████████"

    local BASE_ARGS=(
        "--config" "${CONFIG_DIR}/default.yaml"
        "--llm_api_url" "${LLM_API_URL}"
        "--output_dir" "${OUTPUT_DIR}"
        "--data_path" "${DATA_PATH}"
        "--preprocessed_path" "${PREPROCESSED_PATH}"
    )

    # 1.1 基础模型（默认配置 + 10 epochs）
    run_cmd "Baseline: 完整模型 (epochs=10)" \
        train.py "${BASE_ARGS[@]}" --epochs 10

    # 1.2 基础模型（更长训练，如有需要可取消注释）
    # run_cmd "Baseline: 完整模型 (epochs=30)" \
    #     train.py "${BASE_ARGS[@]}" --epochs 30

    # 1.3 不同 pos_weight 对比（可选）
    # run_cmd "Baseline: pos_weight=3.0" \
    #     train.py "${BASE_ARGS[@]}" --name "aml_baseline_pos3" --pos_weight 3.0
    # run_cmd "Baseline: pos_weight=7.0" \
    #     train.py "${BASE_ARGS[@]}" --name "aml_baseline_pos7" --pos_weight 7.0

    echo ""
    echo "  [基础实验完成]"
}

# ============================================================
# 2. 消融实验 (Ablation Study)
# ============================================================
run_ablation() {
    echo ""
    echo "████████████████████████████████████████████████████████████████████"
    echo "  消融实验 (Ablation Study) - 去除各模块"
    echo "████████████████████████████████████████████████████████████████████"

    local BASE_ARGS=(
        "--config" "${CONFIG_DIR}/default.yaml"
        "--llm_api_url" "${LLM_API_URL}"
        "--output_dir" "${OUTPUT_DIR}"
        "--data_path" "${DATA_PATH}"
        "--preprocessed_path" "${PREPROCESSED_PATH}"
    )

    # 方法一：直接使用 ablation 配置文件的简化消融
    # 注意：这种方法使用 runner.py 内部的 variant_map 自动处理所有消融变体
    run_cmd "消融实验: 批量运行所有变体 (通过 runner._run_ablation)" \
        train.py "${BASE_ARGS[@]}" --ablation --name "ablation_all"

    # 方法二：逐一运行每个消融变体（更清晰，可单独观察每个结果）
    local ABLATION_CONFIGS=(
        "${CONFIG_DIR}/ablation/wo_ca1.yaml"
        "${CONFIG_DIR}/ablation/wo_ca3.yaml"
        "${CONFIG_DIR}/ablation/wo_mpfc.yaml"
        "${CONFIG_DIR}/ablation/wo_vta.yaml"
        "${CONFIG_DIR}/ablation/wo_llm.yaml"
    )

    for config in "${ABLATION_CONFIGS[@]}"; do
        local name
        name=$(basename "${config}" .yaml)
        run_cmd "消融实验: ${name}" \
            train.py "${BASE_ARGS[@]}" \
            --config "${config}" \
            --name "ablation_${name}"
    done

    echo ""
    echo "  [消融实验完成]"
}

# ============================================================
# 3. 参数敏感性实验 (Sensitivity Analysis)
# ============================================================
run_sensitivity() {
    echo ""
    echo "████████████████████████████████████████████████████████████████████"
    echo "  参数敏感性实验 (Sensitivity Analysis)"
    echo "████████████████████████████████████████████████████████████████████"

    local BASE_ARGS=(
        "--llm_api_url" "${LLM_API_URL}"
        "--output_dir" "${OUTPUT_DIR}"
        "--data_path" "${DATA_PATH}"
        "--preprocessed_path" "${PREPROCESSED_PATH}"
    )

    local SWEEP_CONFIGS=(
        "${CONFIG_DIR}/sensitivity/batch_size.yaml"
        "${CONFIG_DIR}/sensitivity/learning_rate.yaml"
        "${CONFIG_DIR}/sensitivity/hidden_dim.yaml"
        "${CONFIG_DIR}/sensitivity/dropout.yaml"
        "${CONFIG_DIR}/sensitivity/pos_weight.yaml"
    )

    for config in "${SWEEP_CONFIGS[@]}"; do
        local name
        name=$(basename "${config}" .yaml)
        run_cmd "参数敏感性: ${name}" \
            train.py "${BASE_ARGS[@]}" \
            --config "${config}" \
            --name "sweep_${name}"
    done

    # 额外扫描：组合参数扫描（可选的深入分析）
    # run_cmd "组合扫描: lr x batch_size" \
    #     train.py "${BASE_ARGS[@]}" \
    #     --config "${CONFIG_DIR}/default.yaml" \
    #     --sweep \
    #     --name "sweep_lr_bs" \
    #     --epochs 5

    echo ""
    echo "  [参数敏感性实验完成]"
}

# ============================================================
# 4. 多随机种子实验 (Multi-Seed)
# ============================================================
run_multi_seed() {
    echo ""
    echo "████████████████████████████████████████████████████████████████████"
    echo "  多随机种子实验 (Multi-Seed) - 评估稳定性"
    echo "████████████████████████████████████████████████████████████████████"

    local BASE_ARGS=(
        "--config" "${CONFIG_DIR}/default.yaml"
        "--llm_api_url" "${LLM_API_URL}"
        "--output_dir" "${OUTPUT_DIR}"
        "--data_path" "${DATA_PATH}"
        "--preprocessed_path" "${PREPROCESSED_PATH}"
    )

    # 4.1 使用 runner 内置的多种子模式（自动统计 mean/std）
    run_cmd "多种子实验: seeds=[42,123,456,789,1111] (通过 runner._run_multi_seed)" \
        train.py "${BASE_ARGS[@]}" --multi_seed --name "multi_seed_5runs"

    # 4.2 逐一指定不同种子（方便单独查看每次运行结果）
    local SEEDS=(42 123 456 789 1111)
    for seed in "${SEEDS[@]}"; do
        run_cmd "多种子实验: seed=${seed}" \
            train.py "${BASE_ARGS[@]}" \
            --seed "${seed}" \
            --name "multi_seed_seed${seed}"
    done

    echo ""
    echo "  [多随机种子实验完成]"
}

# ============================================================
# 5. 汇总结果
# ============================================================
summarize_results() {
    echo ""
    echo "████████████████████████████████████████████████████████████████████"
    echo "  实验结果汇总"
    echo "████████████████████████████████████████████████████████████████████"
    echo ""
    echo "输出目录: ${OUTPUT_DIR}"
    echo ""

    if $DRY_RUN; then
        echo "  (DRY RUN - 未生成实际结果)"
        return
    fi

    # 统计各实验的 results.csv
    echo "--- 各实验结果文件 ---"
    find "${OUTPUT_DIR}" -name "results.csv" -type f 2>/dev/null | sort || echo "  (无结果文件)"
    echo ""

    # 统计实验数量
    local exp_count
    exp_count=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d 2>/dev/null | wc -l)
    exp_count=$((exp_count - 1))  # 减去 outputs 自身
    echo "实验子目录数: ${exp_count}"

    # 列出所有实验目录及其最新 best_val_f1
    echo ""
    echo "--- 各实验 best_val_f1 概览 ---"
    find "${OUTPUT_DIR}" -maxdepth 1 -type d ! -path "${OUTPUT_DIR}" 2>/dev/null | sort | while read -r exp_dir; do
        local exp_name
        exp_name=$(basename "${exp_dir}")
        local csv_file="${exp_dir}/results.csv"
        if [ -f "${csv_file}" ]; then
            # 提取 best_val_f1（跳过 header，取最后一列）
            local f1
            f1=$(tail -1 "${csv_file}" | cut -d',' -f7 2>/dev/null || echo "N/A")
            echo "  ${exp_name}: best_val_f1 = ${f1}"
        else
            echo "  ${exp_name}: (results.csv not found)"
        fi
    done

    echo ""
    echo "=============================="
    echo "  全部实验运行结束"
    echo "  详细结果请查看各实验目录下的 results.csv"
    echo "=============================="
}

# ============================================================
# 主流程
# ============================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║          AML 反洗钱检测 · 全面实验脚本                          ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  LLM API : ${LLM_API_URL}"
    echo "║  输出目录: ${OUTPUT_DIR}"
    echo "║  数据路径: ${DATA_PATH}"
    echo "║  Dry Run : ${DRY_RUN}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""

    local start_time
    start_time=$(date +%s)

    if $RUN_BASELINE; then
        run_baseline
    fi

    if $RUN_ABLATION; then
        run_ablation
    fi

    if $RUN_SENSITIVITY; then
        run_sensitivity
    fi

    if $RUN_MULTI_SEED; then
        run_multi_seed
    fi

    summarize_results

    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "总耗时: $((elapsed / 60)) 分 $((elapsed % 60)) 秒"
    echo ""
}

main
