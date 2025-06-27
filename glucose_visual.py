import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from collections import deque
import re
import sys
import os
import logging
import copy

# ------------------------------
# 配置日志
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# 辅助函数
# ------------------------------

def initialize_session_state():
    """初始化会话状态"""
    if 'params' not in st.session_state:
        st.session_state.params = initial_params.copy()
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()
    if 'single_slope' not in st.session_state:
        st.session_state.single_slope = None
    if 'slope_source' not in st.session_state:
        st.session_state.slope_source = ""
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'active_formulas' not in st.session_state:
        st.session_state.active_formulas = ['formula1']
    if 'shift_value' not in st.session_state:
        st.session_state.shift_value = 10
    if 'active_modules' not in st.session_state:
        st.session_state.active_modules = []
    if 'yaxis_range' not in st.session_state:
        st.session_state.yaxis_range = {
            'current_min': -1.0,
            'current_max': 10.0,
            'glucose_min': 0,
            'glucose_max': 27
        }
    if 'chart_config' not in st.session_state:
        st.session_state.chart_config = {
            'raw_current': {'show': True, 'color': '#636EFA', 'width': 1.0, 'dash': 'solid'},
            'filtered_current': {'show': True, 'color': '#00B5F7', 'width': 1.0, 'dash': 'solid'},
            'decay_compensated_current': {'show': True, 'color': '#FFA15A', 'width': 1.0, 'dash': 'solid'},
            'depression_compensated_current': {'show': True, 'color': '#B82E2E', 'width': 1.0, 'dash': 'solid'},
            'bgm_points': {'show': True, 'color': '#FF6692', 'width': 8.0},
            'formula_original': {'show': True, 'width': 1.0, 'dash': 'dot'},
            'formula_adjusted': {'show': True, 'width': 3.0, 'dash': 'solid'},
        }
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'current_tab_index' not in st.session_state:
        st.session_state.current_tab_index = 0
    if 'file_params' not in st.session_state:
        st.session_state.file_params = {}
    if 'formula_colors' not in st.session_state:
        st.session_state.formula_colors = {
            'formula1': '#00CC96',
            'formula2': '#636EFA',
            'formula3': '#EF553B',
            'formula4': '#AB63FA'
        }


# ------------------------------
# 血糖计算公式
# ------------------------------

def calculate_glucose1(current, slope, a, b, c, d):
    """公式1: current * (a * slope² - b * slope + c) + d"""
    return current * (a * slope ** 2 - b * slope + c) + d


def calculate_glucose2(current, slope, a, b, c, d):
    """公式2: current * ( slope * a + b) * c + d"""
    return current * (slope * a + b) * c + d


def calculate_glucose3(current, slope, a, b, c, d):
    """公式3: a * current + b * slope + c"""
    return (current + d) * (slope * a + b) * c


def calculate_glucose4(current, slope, intercept):
    """公式4: a * current * slope + b"""
    return ((current + intercept) / (slope / 100)) / 18


# 初始参数值
initial_params = {
    'formula1': {
        'a': 1.631893041722917,
        'b': 8.254169802443963,
        'c': 12.436342765983326,
        'd': 2.5,
    },
    'formula2': {
        'a': 1.0,
        'b': 1.0,
        'c': 1.0,
        'd': 1.0,
    },
    'formula3': {
        'a': 1.0,
        'b': 1.0,
        'c': 1.0,
        'd': 1.0,
    },
    'formula4': {
        'intercept': 1.0,
    }
}


# ------------------------------
# 信号处理函数
# ------------------------------

def clean_current_signal(current, extreme_threshold=600, base_threshold=0.1,
                         kalman_R=0.5, kalman_Q=0.02):
    """
    对电流信号进行清洗，依次去除极端值、低频噪声、高频噪声

    参数:
        current: 输入电流序列（numpy 数组或列表）
        extreme_threshold: 极端值阈值（超过该值的将被替换）
        base_threshold: 低频噪声的基础比值阈值
        kalman_R: 卡尔曼滤波器的观测噪声协方差
        kalman_Q: 卡尔曼滤波器的过程噪声协方差

    返回:
        清洗后的电流序列（numpy 数组）
    """
    try:
        if KalmanFilter is None:
            st.warning("卡尔曼滤波器不可用，跳过电流过滤")
            return np.array(current, dtype=float)

        current = np.array(current, dtype=float)

        # Step 1: 去除极端异常值
        current_cleaned = current.copy()
        current_cleaned[current_cleaned > extreme_threshold] = 0

        # Step 2: 去除低频噪声
        startpoint = 120
        current_filtered = current_cleaned.copy()
        for i in range(len(current_filtered)):
            if i > startpoint:
                dynamic_threshold = 1.0 * np.std(current_filtered[max(0, i - 25):i])
                rate_threshold = base_threshold + dynamic_threshold
                window_24 = current_filtered[max(0, i - 24):i]
                window_2 = current_filtered[max(0, i - 2):i]

                if current_filtered[i] > 10 or current_filtered[i] < 0.1:
                    current_filtered[i] = np.mean(window_24)
                elif current_filtered[i] > (1 + 1.5 * rate_threshold) * np.mean(window_2) or \
                        current_filtered[i] < (1 - rate_threshold) * np.mean(window_2):
                    current_filtered[i] = np.mean(window_2)
                    if np.std(current_filtered[max(0, i - 5):i]) < 0.005:
                        current_filtered[i] = 0.6 * current_cleaned[i] + 0.4 * current_filtered[i - 1]

        # Step 3: 卡尔曼滤波去除高频噪声
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([current_filtered[0]])  # 初始状态
        kf.F = np.array([[1]])  # 状态转移矩阵
        kf.H = np.array([[1]])  # 观测矩阵
        kf.P *= 1000  # 初始协方差
        kf.R = kalman_R  # 观测噪声协方差
        kf.Q = kalman_Q  # 过程噪声协方差

        filtered_output = []
        for i in range(len(current_filtered)):
            kf.predict()
            kf.update(current_filtered[i])
            filtered_output.append(kf.x[0])

        return np.array(filtered_output)

    except Exception as e:
        logger.error(f"电流信号清洗失败: {str(e)}")
        return np.array(current, dtype=float)


def compensate_decay(current, comrate=0.75, A=0.5444, k=0.1561, B=0.3473,
                     start_index=8640, long_window=5 * 24 * 60, short_window=24 * 60,
                     detect_interval=30, slope_thresh=-0.001, mean_change_thresh=-0.15,
                     trigger_count=1):
    """
    衰减补偿函数

    参数:
        current: 原始电流信号（1D array）
        其他参数: 衰减模型参数和检测参数

    返回:
        补偿后信号，与输入同形
    """
    try:
        compensated = current.copy()
        buffer = deque(maxlen=long_window)  # 长期数据缓冲区
        trigger_counter = 0  # 连续触发计数器
        compensation_active = False  # 补偿激活标志
        last_detect_index = -detect_interval  # 上次检测位置初始化

        for idx in range(len(current)):
            # 更新缓冲区
            buffer.append(current[idx])

            # 执行检测的条件
            if (idx >= start_index and (idx - last_detect_index) >= detect_interval):
                # 获取窗口数据
                long_data = list(buffer)[-long_window:]
                short_data = list(buffer)[-short_window:]

                # 执行衰减检测
                if len(short_data) < 2 or len(long_data) < 2:
                    continue

                slope = linregress(range(len(short_data)), short_data).slope
                mean_change = (np.mean(short_data) - np.mean(long_data)) / np.mean(long_data)

                if mean_change < mean_change_thresh:
                    trigger_counter += 1
                    if trigger_counter >= trigger_count:
                        compensation_active = True
                        trigger_counter = 0  # 重置计数器
                else:
                    trigger_counter = 0
                    compensation_active = False

                last_detect_index = idx

            # 执行补偿计算
            if compensation_active and idx >= start_index:
                t = (idx - start_index) / 1440.0  # 转换为天
                decay_factor = A * np.exp(-k * t) + B
                decay_factor = max(decay_factor, 1e-6)  # 安全阈值
                compensated[idx] = comrate * (current[idx] / decay_factor - current[idx]) + current[idx]

        return compensated

    except Exception as e:
        logger.error(f"衰减补偿失败: {str(e)}")
        return current.copy()


def compensate_depression(glucose_judge_hole, compensated_current,
                          window_size=8, drop_threshold=1.4,
                          max_compensation_duration=120, baseline_diff_threshold=1.0,
                          base_compensation_ratio=0.2):
    """
    凹陷信号补偿函数（含缓慢过渡阶段）

    参数:
        glucose_judge_hole: 用于判断凹陷的血糖曲线
        compensated_current: 需要补偿的目标电流曲线

    返回:
        凹陷补偿后的电流曲线
    """
    try:
        compensated = compensated_current.copy()
        compensation_active = False  # 正常补偿阶段
        in_transition = False  # 处于缓慢过渡阶段
        compensation_start = 0  # 正常补偿开始时索引
        transition_start = 0  # 过渡阶段开始时索引
        transition_target = 0  # 过渡阶段目标时长（等于正常补偿持续的时间）
        baseline_value = 0  # 下降前基准值

        for i in range(max(window_size + 5, 120), len(compensated)):
            # 检测正常补偿阶段启动条件
            if not compensation_active and not in_transition:
                start_idx = i - window_size
                start_value = glucose_judge_hole[start_idx]
                end_value = glucose_judge_hole[i]
                drop_amount = start_value - end_value

                # 判断窗口内是否严格连续下降
                continuous_descent = all(
                    glucose_judge_hole[j] > glucose_judge_hole[j + 1]
                    for j in range(i - window_size, i)
                )

                # 判断窗口起始点与前5个点均值的差异
                baseline_stable = True
                prev_values = glucose_judge_hole[i - window_size - 5: i - window_size]
                if len(prev_values) > 0:
                    prev_avg = np.mean(prev_values)
                    if (start_value - prev_avg) > baseline_diff_threshold:
                        baseline_stable = False

                # 判断该点处于峰值上正常降糖阶段还是平稳期的凹陷阶段
                hole_phase = True
                hist_region = glucose_judge_hole[:start_idx]
                if len(hist_region) > 0:
                    hist_max = min(np.max(hist_region), 20.0)
                    hole_phase = start_value < 0.7 * hist_max

                # 满足所有条件时，启动正常补偿阶段
                if drop_amount >= drop_threshold and continuous_descent and baseline_stable and hole_phase:
                    compensation_active = True
                    compensation_start = i
                    baseline_value = compensated[i - window_size]  # 记录下降前的基准值
                    baseline_value_judge = start_value  # 记录下降前的用于判断的基准值

            # 如果处于正常补偿阶段
            if compensation_active:
                compensation_duration = i - compensation_start
                frac = min((compensation_duration + 1.0) / 5.0, 1.0)
                eff_ratio = 1.0 - (1.0 - base_compensation_ratio) * frac

                # 检测是否满足转换到缓慢过渡阶段的条件
                current_value = glucose_judge_hole[i]
                if current_value >= (
                        baseline_value_judge - 0.4 * drop_threshold) or compensation_duration >= max_compensation_duration:
                    # 进入缓慢过渡阶段
                    in_transition = True
                    transition_start = i
                    transition_target = compensation_duration
                    compensation_active = False  # 结束正常补偿阶段

                # 正常补偿：保持固定补偿比例
                if compensation_active:
                    compensated[i] = baseline_value - (baseline_value - compensated[i]) * eff_ratio

            # 如果处于缓慢过渡阶段
            if in_transition and not compensation_active:
                transition_progress = (i - transition_start) / transition_target
                if transition_progress >= 1.0:
                    in_transition = False
                    continue

                # 补偿比例随进度线性递减
                gradual_ratio = base_compensation_ratio + (1 - base_compensation_ratio) * transition_progress
                compensated[i] = baseline_value - (baseline_value - compensated[i]) * gradual_ratio

        return compensated

    except Exception as e:
        logger.error(f"凹陷补偿失败: {str(e)}")
        return compensated_current.copy()


def compensate_stable_baseline(glucose, day_length=1440, stable_start_day=3, stable_end_day=6):
    """
    血糖补正模块（逐点处理）

    输入:
         glucose: 1D numpy 数组，通常传入 compensated_hole_glucose
    参数:
        day_length: 每天采样点数，默认 1440（1min/点）
        stable_start_day: 用于计算"稳定基准"a的起始天（1 起始）
        stable_end_day: 用于计算"稳定基准"a的结束天（含）

    返回:
        1D numpy 数组，同长度，经稳定基准补偿后的血糖
    """
    try:
        compensated = glucose.copy()
        start_idx = (stable_start_day - 1) * day_length
        end_idx = stable_end_day * day_length

        if start_idx >= len(glucose) or end_idx > len(glucose):
            st.warning(f"稳定基准补偿: 索引超出范围 ({start_idx}-{end_idx})，跳过此步骤")
            return glucose

        stable_day_val = np.mean(glucose[start_idx:end_idx])
        yesterday = deque(maxlen=day_length)

        for i in range(len(glucose)):
            if i > 0:
                yesterday.append(glucose[i - 1])

            if i >= end_idx and len(yesterday) == day_length:
                stable_yesterday_val = np.mean(yesterday)
                compensated[i] = glucose[i] - stable_yesterday_val + stable_day_val

        return compensated

    except Exception as e:
        logger.error(f"稳定基线补偿失败: {str(e)}")
        return glucose.copy()


# ------------------------------
# 数据评估函数
# ------------------------------

def calculate_mard(sensor_glucose, bgm, shift=10):
    """
    计算MARD值

    参数:
        sensor_glucose: CGM 得到的血糖曲线
        bgm: 参考的 BGM 值
        shift: 用于比较的时间偏移（单位：数据点索引偏移）

    返回:
        平均相对差异（百分比）和平均绝对差异
    """
    try:
        # 找出有效的 BGM 位置（非 NaN）
        valid_idx = ~np.isnan(bgm)

        # 如果没有有效点，返回NaN
        if not np.any(valid_idx):
            return np.nan, np.nan

        # 获取有效BGM的索引位置
        bgm_idx = np.where(valid_idx)[0]

        # 计算偏移后的索引位置
        shifted_idx = bgm_idx + shift

        # 过滤越界情况
        valid_shift = (shifted_idx >= 0) & (shifted_idx < len(sensor_glucose))
        bgm_idx_valid = bgm_idx[valid_shift]
        shifted_idx_valid = shifted_idx[valid_shift]

        # 获取有效数据
        bgm_values = bgm[bgm_idx_valid]
        sensor_values = sensor_glucose[shifted_idx_valid]

        # 如果有效数据为空，返回NaN
        if len(bgm_values) == 0:
            return np.nan, np.nan

        # 计算相对差异和绝对差异
        relative_diff = np.abs(sensor_values - bgm_values) / bgm_values
        absolute_diff = np.abs(sensor_values - bgm_values)

        # 返回平均相对差异（百分比）和平均绝对差异
        return np.mean(relative_diff) * 100, np.mean(absolute_diff)

    except Exception as e:
        logger.error(f"MARD计算失败: {str(e)}")
        return np.nan, np.nan


# ------------------------------
# 数据处理辅助函数
# ------------------------------

def detect_column_names(df):
    """根据常见名称模式自动检测列名"""
    mapping = {}

    # 检测电流列名
    current_patterns = ['WE1 Current', '电流值1', '电流', 'current', 'Current']
    for pattern in current_patterns:
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                mapping['current'] = col
                break
        if 'current' in mapping:
            break

    # 检测BGM列名
    bgm_patterns = ['BGM', '指尖血数据', '参考值', '参考血糖']
    for pattern in bgm_patterns:
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                mapping['bgm'] = col
                break
        if 'bgm' in mapping:
            break

    # 检测slope列名
    slope_patterns = ['Sen', 'Slope', '斜率', 'sens']
    for pattern in slope_patterns:
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                mapping['slope'] = col
                break
        if 'slope' in mapping:
            break

    # 检测序列号列名
    serial_patterns = ['Serial Number', 'Serial', '序号', '编号']
    for pattern in serial_patterns:
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                mapping['serial'] = col
                break
        if 'serial' in mapping:
            break

    return mapping


def convert_to_numeric(df, column):
    """将指定列转换为数值类型，处理转换错误"""
    if column in df.columns:
        # 尝试转换为数值类型
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # 检查并报告转换错误
        if df[column].isnull().any():
            num_errors = df[column].isnull().sum()
            st.warning(f"在列 '{column}' 中发现 {num_errors} 个无法转换为数值的值，已替换为NaN")
    return df


# ------------------------------
# 可视化函数
# ------------------------------

def create_glucose_chart(data, file_name):
    """创建血糖浓度分析图表 - 修改为接受特定数据"""
    if data.empty or not st.session_state.active_formulas:
        return None

    # 确保数据按Serial Number排序
    if 'Serial Number' in data.columns:
        sorted_data = data.sort_values(by='Serial Number')
        x_values = sorted_data['Serial Number']
        x_title = 'Serial Number'
    else:
        sorted_data = data
        x_values = sorted_data.index
        x_title = '数据点索引'

    # 创建图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    config = st.session_state.chart_config

    # 添加电流系列
    add_current_traces(fig, x_values, sorted_data, config)

    # 添加BGM点图
    add_bgm_points(fig, sorted_data, config)

    # 添加血糖曲线
    add_glucose_traces(fig, x_values, sorted_data, config, file_name)

    # 设置图表布局
    fig.update_layout(
        title=f'血糖浓度分析: {os.path.basename(file_name)}',
        xaxis_title=x_title,
        yaxis_title='血糖浓度 (mg/dL)',
        yaxis2_title='电流值 (μA)',
        height=600,
        autosize=True,
        legend=dict(x=0, y=1.1, orientation='h'),
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # 设置Y轴范围
    fig.update_yaxes(
        range=[st.session_state.yaxis_range['current_min'],
               st.session_state.yaxis_range['current_max']],
        secondary_y=True
    )

    fig.update_yaxes(
        range=[st.session_state.yaxis_range['glucose_min'],
               st.session_state.yaxis_range['glucose_max']],
        secondary_y=False
    )

    return fig


def add_current_traces(fig, x_values, data, config):
    """添加电流系列到图表"""
    # 原始电流
    if config['raw_current']['show'] and 'WE1 Current' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=data['WE1 Current'],
                mode='lines',
                name='原始电流',
                line=dict(
                    color=config['raw_current']['color'],
                    width=config['raw_current']['width'],
                    dash=config['raw_current']['dash']
                ),
                opacity=0.7
            ),
            secondary_y=True
        )

        # 过滤后电流 - 仅在模块激活时显示
    if ('Filtered Current' in data.columns and
            'current_filter' in st.session_state.active_modules and
            config['filtered_current']['show']):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=data['Filtered Current'],
                mode='lines',
                name='过滤后电流',
                line=dict(
                    color=config['filtered_current']['color'],
                    width=config['filtered_current']['width'],
                    dash=config['filtered_current']['dash']
                ),
                opacity=0.7,
                visible=True
            ),
            secondary_y=True
        )

        # 衰减补偿电流 - 仅在模块激活时显示
    if ('Decay Compensated Current' in data.columns and
            'decay_compensation' in st.session_state.active_modules and
            config['decay_compensated_current']['show']):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=data['Decay Compensated Current'],
                mode='lines',
                name='衰减补偿电流',
                line=dict(
                    color=config['decay_compensated_current']['color'],
                    width=config['decay_compensated_current']['width'],
                    dash=config['decay_compensated_current']['dash']
                ),
                opacity=0.7,
                visible=True
            ),
            secondary_y=True
        )

        # 凹陷补偿电流 - 仅在模块激活时显示
    if ('Depression Compensated Current' in data.columns and
            'depression_compensation' in st.session_state.active_modules and
            config['depression_compensated_current']['show']):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=data['Depression Compensated Current'],
                mode='lines',
                name='凹陷补偿电流',
                line=dict(
                    color=config['depression_compensated_current']['color'],
                    width=config['depression_compensated_current']['width'],
                    dash=config['depression_compensated_current']['dash']
                ),
                opacity=0.7,
                visible=True
            ),
            secondary_y=True
        )


def add_bgm_points(fig, data, config):
    """添加BGM点到图表"""
    if config['bgm_points']['show'] and 'BGM' in data.columns:
        bgm_data = data.dropna(subset=['BGM'])
        if not bgm_data.empty:
            x_values = bgm_data['Serial Number'] if 'Serial Number' in bgm_data else bgm_data.index
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=bgm_data['BGM'],
                    mode='markers',
                    name='参考血糖值',
                    marker=dict(
                        color=config['bgm_points']['color'],
                        size=config['bgm_points']['width']
                    )
                ),
                secondary_y=False
            )


def add_glucose_traces(fig, x_values, data, config, file_name):
    """添加血糖曲线到图表"""
    # 获取该文件的参数
    file_params = st.session_state.file_params.get(file_name, st.session_state.params)
    formula_colors = st.session_state.formula_colors

    for formula in st.session_state.active_formulas:
        if formula not in data.columns:
            continue

        formula_name = formula_options[formula].split(':')[0]
        trace_name = f'{formula_name} (原始)'
        adjusted_name = f'{formula_name} (调整)'

        # 获取该公式的颜色
        formula_color = formula_colors.get(formula, '#00CC96')

        # 原始血糖曲线
        if config['formula_original']['show']:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=data[formula],
                    mode='lines',
                    name=trace_name,
                    line=dict(
                        color=formula_color,
                        width=config['formula_original']['width'],
                        dash=config['formula_original']['dash']
                    ),
                    opacity=0.7
                ),
                secondary_y=False
            )

        # 调整后的血糖曲线
        if config['formula_adjusted']['show'] and f'{formula}_adjusted' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=data[f'{formula}_adjusted'],
                    mode='lines',
                    name=adjusted_name,
                    line=dict(
                        color=formula_color,
                        width=config['formula_adjusted']['width'],
                        dash=config['formula_adjusted']['dash']
                    )
                ),
                secondary_y=False
            )


# ------------------------------
# 数据处理函数
# ------------------------------

def process_uploaded_files(uploaded_files):
    """处理上传的文件并加载数据"""
    all_data = []
    for file in uploaded_files:
        try:
            df = pd.read_excel(file)
            column_mapping = detect_column_names(df)
            st.session_state.column_mapping = column_mapping

            # 检查必需的列是否存在
            required_cols = ['current', 'bgm']
            if not all(col in column_mapping for col in required_cols):
                missing = [col for col in required_cols if col not in column_mapping]
                st.warning(f"文件 {file.name} 缺少必需的列: {', '.join(missing)}，已跳过")
                continue

            current_col = column_mapping['current']
            bgm_col = column_mapping['bgm']

            # 将关键列转换为数值类型
            df = convert_to_numeric(df, current_col)
            df = convert_to_numeric(df, bgm_col)

            # 处理slope列
            slope_col = handle_slope_column(df, column_mapping, file.name)

            # 确保重命名为标准列名
            df.rename(columns={current_col: 'WE1 Current'}, inplace=True)
            df.rename(columns={bgm_col: 'BGM'}, inplace=True)

            if slope_col:
                df.rename(columns={slope_col: 'slope'}, inplace=True)

            # 处理序列号
            handle_serial_number(df, column_mapping)

            df['Source'] = file.name

            # 初始化该文件的参数
            if file.name not in st.session_state.file_params:
                st.session_state.file_params[file.name] = copy.deepcopy(initial_params)

            all_data.append(df)

        except Exception as e:
            st.error(f"读取文件 {file.name} 时出错: {str(e)}")
            logger.exception(f"文件处理错误: {file.name}")

    return all_data


def handle_slope_column(df, column_mapping, filename):
    """处理slope列"""
    slope_col = None
    if 'slope' in column_mapping:
        slope_col = column_mapping['slope']
        df = convert_to_numeric(df, slope_col)
        valid_slopes = df[slope_col].dropna()

        if len(valid_slopes) == 1:
            single_slope = valid_slopes.iloc[0]
            df[slope_col] = single_slope
            st.session_state.single_slope = single_slope
            st.session_state.slope_source = filename
            st.info(f"文件 {filename} 中检测到单个slope值: {single_slope:.4f}，已应用于所有数据点")
        elif len(valid_slopes) == 0:
            st.warning(f"文件 {filename} 中没有有效的slope值")
        else:
            st.session_state.single_slope = None
    else:
        if st.session_state.single_slope is not None:
            df['slope'] = st.session_state.single_slope
            st.info(f"文件 {filename} 中没有检测到slope列，使用之前设置的slope值: {st.session_state.single_slope:.4f}")
            slope_col = 'slope'
        else:
            st.warning(f"文件 {filename} 没有检测到slope列，且没有可用的单一slope值")
    return slope_col


def handle_serial_number(df, column_mapping):
    """处理序列号列"""
    if 'serial' in column_mapping:
        serial_col = column_mapping['serial']
        df = convert_to_numeric(df, serial_col)
        df.rename(columns={serial_col: 'Serial Number'}, inplace=True)
        # 修复弃用警告：使用ffill()和bfill()替代fillna(method='...')
        df['Serial Number'] = df['Serial Number'].ffill().bfill()
    else:
        df['Serial Number'] = range(1, len(df) + 1)


def apply_processing_modules(data, file_name):
    """应用选定的处理模块到数据"""
    # 获取该文件的参数
    file_params = st.session_state.file_params.get(file_name, st.session_state.params)

    processed_data = data.copy()

    # 电流过滤模块
    if 'current_filter' in st.session_state.active_modules:
        try:
            processed_data['Filtered Current'] = clean_current_signal(
                processed_data['WE1 Current'].values
            )
            st.success("电流过滤模块应用成功")
        except Exception as e:
            st.error(f"电流过滤失败: {str(e)}")
            processed_data['Filtered Current'] = processed_data['WE1 Current']
    else:
        processed_data['Filtered Current'] = processed_data['WE1 Current']

    # 衰减补偿模块
    if 'decay_compensation' in st.session_state.active_modules:
        try:
            processed_data['Decay Compensated Current'] = compensate_decay(
                processed_data['Filtered Current'].values
            )
            st.success("衰减补偿模块应用成功")
        except Exception as e:
            st.error(f"衰减补偿失败: {str(e)}")
            processed_data['Decay Compensated Current'] = processed_data['Filtered Current']
    else:
        processed_data['Decay Compensated Current'] = processed_data['Filtered Current']

    # 计算初始血糖值用于凹陷补偿
    if 'slope' in processed_data.columns:
        calculate_initial_glucose(processed_data, file_params)

    # 凹陷补偿模块
    if 'depression_compensation' in st.session_state.active_modules and 'Initial Glucose' in processed_data:
        try:
            processed_data['Depression Compensated Current'] = compensate_depression(
                processed_data['Initial Glucose'].values,
                processed_data['Decay Compensated Current'].values
            )
            st.success("凹陷补偿模块应用成功")
        except Exception as e:
            st.error(f"凹陷补偿失败: {str(e)}")
            processed_data['Depression Compensated Current'] = processed_data['Decay Compensated Current']
    else:
        processed_data['Depression Compensated Current'] = processed_data['Decay Compensated Current']

    # 计算最终血糖浓度
    if 'slope' in processed_data.columns:
        calculate_final_glucose(processed_data, file_params)

    return processed_data


def calculate_initial_glucose(data, params):
    """计算初始血糖值"""
    for formula in st.session_state.active_formulas:
        if formula == 'formula1':
            data['Initial Glucose'] = calculate_glucose1(
                data['Decay Compensated Current'],
                data['slope'],
                params[formula]['a'],
                params[formula]['b'],
                params[formula]['c'],
                params[formula]['d']
            )
            break  # 只使用第一个公式计算初始血糖
        elif formula == 'formula2':
            data['Initial Glucose'] = calculate_glucose2(
                data['Decay Compensated Current'],
                data['slope'],
                params[formula]['a'],
                params[formula]['b'],
                params[formula]['c'],
                params[formula]['d']
            )
            break
        elif formula == 'formula3':
            data['Initial Glucose'] = calculate_glucose3(
                data['Decay Compensated Current'],
                data['slope'],
                params[formula]['a'],
                params[formula]['b'],
                params[formula]['c'],
                params[formula]['d']
            )
            break
        elif formula == 'formula4':
            data['Initial Glucose'] = calculate_glucose4(
                data['Decay Compensated Current'],
                data['slope'],
                params[formula]['intercept']
            )
            break


def calculate_final_glucose(data, params):
    """计算最终血糖浓度"""
    for formula in st.session_state.active_formulas:
        if formula == 'formula1':
            data[formula] = calculate_glucose1(
                data['Depression Compensated Current'],
                data['slope'],
                params[formula]['a'],
                params[formula]['b'],
                params[formula]['c'],
                params[formula]['d']
            )
        elif formula == 'formula2':
            data[formula] = calculate_glucose2(
                data['Depression Compensated Current'],
                data['slope'],
                params[formula]['a'],
                params[formula]['b'],
                params[formula]['c'],
                params[formula]['d']
            )
        elif formula == 'formula3':
            data[formula] = calculate_glucose3(
                data['Depression Compensated Current'],
                data['slope'],
                params[formula]['a'],
                params[formula]['b'],
                params[formula]['c'],
                params[formula]['d']
            )
        elif formula == 'formula4':
            data[formula] = calculate_glucose4(
                data['Depression Compensated Current'],
                data['slope'],
                params[formula]['intercept'],
            )

        # 应用衰减基线调整
        if 'baseline_adjustment' in st.session_state.active_modules:
            data[f'{formula}_adjusted'] = compensate_stable_baseline(
                data[formula].values
            )
            st.success(f"衰减基线调整模块应用于公式 {formula}")
        else:
            data[f'{formula}_adjusted'] = data[formula]

        # 确保计算结果是数值类型
        data = convert_to_numeric(data, f'{formula}_adjusted')


# ------------------------------
# Streamlit 应用主函数
# ------------------------------

def main():
    # 初始化会话状态
    initialize_session_state()

    # 页面设置
    st.set_page_config(layout="wide", page_title="血糖浓度曲线分析", page_icon="📈")
    st.title("📈 血糖浓度曲线分析")
    st.write("上传Excel文件，获取血糖曲线")

    # ------------------------------
    # 文件上传区域
    # ------------------------------
    with st.expander("📤 上传Excel文件", expanded=True):
        uploaded_files = st.file_uploader(
            "选择Excel文件（支持多个文件）",
            type=["xlsx", "xls"],
            accept_multiple_files=True
        )

    # 处理上传的文件
    if uploaded_files:
        all_data = process_uploaded_files(uploaded_files)

        if all_data:
            processed_files = []
            for df in all_data:
                file_name = df['Source'].iloc[0]

                # 确保slope列存在
                if 'slope' not in df.columns and st.session_state.single_slope is not None:
                    df['slope'] = st.session_state.single_slope

                # 应用处理模块
                processed_data = apply_processing_modules(df, file_name)
                processed_files.append((file_name, processed_data))

            st.session_state.processed_files = processed_files

    # ------------------------------
    # 手动设置slope值
    # ------------------------------
    with st.expander("⚙️ 手动设置单一slope值", expanded=False):
        if st.session_state.single_slope is not None:
            default_slope = st.session_state.single_slope
        else:
            default_slope = 1.0

        new_slope = st.number_input(
            "设置全局slope值（将应用于所有数据点）",
            value=default_slope,
            step=0.01,
            format="%.4f"
        )

        if st.button("应用单一slope值"):
            st.session_state.single_slope = new_slope
            st.success(f"已设置全局slope值为: {new_slope:.4f}")

            # 更新所有文件数据中的slope值
            if st.session_state.processed_files:
                updated_files = []
                for file_name, data in st.session_state.processed_files:
                    data['slope'] = new_slope
                    # 重新计算所有激活公式的血糖浓度
                    recalculate_glucose_for_data(data, file_name)
                    updated_files.append((file_name, data))
                st.session_state.processed_files = updated_files
                st.experimental_rerun()

    # ------------------------------
    # 侧边栏配置
    # ------------------------------
    configure_sidebar()

    # ------------------------------
    # 主界面布局 - 多文件标签页
    # ------------------------------
    if st.session_state.processed_files:
        # 创建标签页
        tab_titles = [f"文件: {os.path.basename(file_name)}" for file_name, _ in st.session_state.processed_files]
        tab_titles.append("汇总")
        tabs = st.tabs(tab_titles)

        # 存储所有文件的MARD结果
        all_mard_results = []

        # 为每个文件显示单独的内容
        for i, (file_name, data) in enumerate(st.session_state.processed_files):
            with tabs[i]:
                col1, col2 = st.columns([4, 1])

                with col1:
                    # 显示该文件的图表
                    fig = create_glucose_chart(data, file_name)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("无法为该文件创建图表")

                    # 显示该文件的数据预览（放在图表下方）
                    with st.expander(f"📊 {os.path.basename(file_name)} 的数据预览", expanded=False):
                        st.dataframe(data.head(10), height=300)

                        # 下载结果按钮
                        csv = data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"下载 {os.path.basename(file_name)} 的结果数据 (CSV)",
                            data=csv,
                            file_name=f'glucose_analysis_{os.path.basename(file_name)}.csv',
                            mime='text/csv'
                        )

                    # 显示该文件的MARD计算
                    st.subheader(f"{os.path.basename(file_name)} 的MARD值计算")
                    mard_results = display_mard_calculation(data, file_name)
                    if mard_results:
                        all_mard_results.extend(mard_results)

                with col2:
                    # 该文件的参数调整面板
                    with st.expander("🧪 参数调整", expanded=True):
                        st.subheader(f"{os.path.basename(file_name)} 的参数设置")
                        file_params = st.session_state.file_params.get(file_name, copy.deepcopy(initial_params))
                        formula_colors = st.session_state.formula_colors

                        # 为每个激活的公式创建参数调整区域
                        for formula in st.session_state.active_formulas:
                            formula_name = formula_options[formula].split(':')[0]

                            st.markdown(f"### {formula_name}参数")

                            # 公式颜色选择器
                            color_key = f"{file_name}_{formula}_color"
                            new_color = st.color_picker(
                                f"{formula_name}曲线颜色",
                                value=formula_colors.get(formula, '#00CC96'),
                                key=color_key
                            )
                            formula_colors[formula] = new_color
                            st.session_state.formula_colors = formula_colors

                            # 根据公式类型创建不同的滑块
                            if formula == 'formula1':
                                # 参数a
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['a'] = st.slider(
                                            "参数 a",
                                            min_value=-5.0,
                                            max_value=5.0,
                                            value=file_params[formula]['a'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_a_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['a'] = st.number_input(
                                            "参数 a (精确值)",
                                            min_value=-5.0,
                                            max_value=5.0,
                                            value=file_params[formula]['a'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_a_input'
                                        )
                                # 参数b
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['b'] = st.slider(
                                            "参数 b",
                                            min_value=-15.0,
                                            max_value=15.0,
                                            value=file_params[formula]['b'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_b_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['b'] = st.number_input(
                                            "参数 b (精确值)",
                                            min_value=-15.0,
                                            max_value=15.0,
                                            value=file_params[formula]['b'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_b_input'
                                        )

                                # 参数c
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['c'] = st.slider(
                                            "参数 c",
                                            min_value=-20.0,
                                            max_value=20.0,
                                            value=file_params[formula]['c'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_c_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['c'] = st.number_input(
                                            "参数 c (精确值)",
                                            min_value=-20.0,
                                            max_value=20.0,
                                            value=file_params[formula]['c'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_c_input'
                                        )

                                # 参数d
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['d'] = st.slider(
                                            "参数 d",
                                            min_value=-10.0,
                                            max_value=10.0,
                                            value=file_params[formula]['d'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_d_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['d'] = st.number_input(
                                            "参数 d (精确值)",
                                            min_value=-10.0,
                                            max_value=10.0,
                                            value=file_params[formula]['d'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_d_input'
                                        )

                            if formula == 'formula2':
                                # 参数a
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['a'] = st.slider(
                                            "参数 a",
                                            min_value=-5.0,
                                            max_value=5.0,
                                            value=file_params[formula]['a'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_a_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['a'] = st.number_input(
                                            "参数 a (精确值)",
                                            min_value=-5.0,
                                            max_value=5.0,
                                            value=file_params[formula]['a'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_a_input'
                                        )
                                # 参数b
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['b'] = st.slider(
                                            "参数 b",
                                            min_value=-15.0,
                                            max_value=15.0,
                                            value=file_params[formula]['b'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_b_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['b'] = st.number_input(
                                            "参数 b (精确值)",
                                            min_value=-15.0,
                                            max_value=15.0,
                                            value=file_params[formula]['b'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_b_input'
                                        )

                                # 参数c
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['c'] = st.slider(
                                            "参数 c",
                                            min_value=-20.0,
                                            max_value=20.0,
                                            value=file_params[formula]['c'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_c_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['c'] = st.number_input(
                                            "参数 c (精确值)",
                                            min_value=-20.0,
                                            max_value=20.0,
                                            value=file_params[formula]['c'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_c_input'
                                        )

                                # 参数d
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['d'] = st.slider(
                                            "参数 d",
                                            min_value=-10.0,
                                            max_value=10.0,
                                            value=file_params[formula]['d'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_d_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['d'] = st.number_input(
                                            "参数 d (精确值)",
                                            min_value=-10.0,
                                            max_value=10.0,
                                            value=file_params[formula]['d'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_d_input'
                                        )

                            if formula == 'formula3':
                                # 参数a
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['a'] = st.slider(
                                            "参数 a",
                                            min_value=-5.0,
                                            max_value=5.0,
                                            value=file_params[formula]['a'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_a_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['a'] = st.number_input(
                                            "参数 a (精确值)",
                                            min_value=-5.0,
                                            max_value=5.0,
                                            value=file_params[formula]['a'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_a_input'
                                        )
                                # 参数b
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['b'] = st.slider(
                                            "参数 b",
                                            min_value=-15.0,
                                            max_value=15.0,
                                            value=file_params[formula]['b'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_b_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['b'] = st.number_input(
                                            "参数 b (精确值)",
                                            min_value=-15.0,
                                            max_value=15.0,
                                            value=file_params[formula]['b'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_b_input'
                                        )

                                # 参数c
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['c'] = st.slider(
                                            "参数 c",
                                            min_value=-20.0,
                                            max_value=20.0,
                                            value=file_params[formula]['c'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_c_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['c'] = st.number_input(
                                            "参数 c (精确值)",
                                            min_value=-20.0,
                                            max_value=20.0,
                                            value=file_params[formula]['c'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_c_input'
                                        )

                                # 参数d
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['d'] = st.slider(
                                            "参数 d",
                                            min_value=-10.0,
                                            max_value=10.0,
                                            value=file_params[formula]['d'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_d_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['d'] = st.number_input(
                                            "参数 d (精确值)",
                                            min_value=-10.0,
                                            max_value=10.0,
                                            value=file_params[formula]['d'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_d_input'
                                        )

                            elif formula == 'formula4':
                                # 参数d
                                with st.container():
                                    col_slider, col_input = st.columns([3, 1])
                                    with col_slider:
                                        file_params[formula]['intercept'] = st.slider(
                                            "参数 intercept",
                                            min_value=-10.0,
                                            max_value=10.0,
                                            value=file_params[formula]['intercept'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_intercept_slider'
                                        )
                                    with col_input:
                                        file_params[formula]['intercept'] = st.number_input(
                                            "参数intercept (精确值)",
                                            min_value=-10.0,
                                            max_value=10.0,
                                            value=file_params[formula]['intercept'],
                                            step=0.001,
                                            format="%.5f",
                                            key=f'{file_name}_{formula}_intercept_input'
                                        )

                            st.write("---")

                        # 保存该文件的参数
                        st.session_state.file_params[file_name] = file_params

                        # 添加重新计算按钮
                        if st.button(f"应用参数并重新计算 {os.path.basename(file_name)}", key=f"recalc_{file_name}"):
                            # 重新计算该文件的血糖曲线
                            recalculate_glucose_for_data(data, file_name)
                            st.session_state.processed_files[i] = (file_name, data)
                            st.experimental_rerun()

                        # 重置按钮
                        if st.button(f"重置参数为初始值", key=f"reset_{file_name}"):
                            st.session_state.file_params[file_name] = copy.deepcopy(initial_params)
                            # 重新计算该文件的血糖曲线
                            recalculate_glucose_for_data(data, file_name)
                            st.session_state.processed_files[i] = (file_name, data)
                            st.experimental_rerun()

        # 汇总标签页
        with tabs[-1]:
            st.header("所有文件汇总")

            if all_mard_results:
                # 创建汇总表格
                summary_data = []
                formula_names = set()

                # 收集所有公式名称
                for result in all_mard_results:
                    formula_names.add(result['公式'])

                # 为每个公式计算平均MARD
                for formula in formula_names:
                    orig_mard_vals = []
                    adj_mard_vals = []

                    for result in all_mard_results:
                        if result['公式'] == formula:
                            orig_mard_vals.append(float(result['原始MARD(%)']))
                            adj_mard_vals.append(float(result['调整后MARD(%)']))

                    if orig_mard_vals and adj_mard_vals:
                        avg_orig = sum(orig_mard_vals) / len(orig_mard_vals)
                        avg_adj = sum(adj_mard_vals) / len(adj_mard_vals)
                        improvement = avg_orig - avg_adj

                        summary_data.append({
                            '公式': formula,
                            '平均原始MARD(%)': f"{avg_orig:.2f}",
                            '平均调整后MARD(%)': f"{avg_adj:.2f}",
                            '平均改进(%)': f"{improvement:.2f}"
                        })

                # 显示汇总表格
                if summary_data:
                    st.subheader("各公式平均MARD值")
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True)

                    # 计算总体平均MARD
                    overall_orig = sum([float(d['平均原始MARD(%)']) for d in summary_data]) / len(summary_data)
                    overall_adj = sum([float(d['平均调整后MARD(%)']) for d in summary_data]) / len(summary_data)
                    overall_improvement = overall_orig - overall_adj

                    st.subheader("总体平均MARD")
                    st.metric("平均原始MARD", f"{overall_orig:.2f}%")
                    st.metric("平均调整后MARD", f"{overall_adj:.2f}%")
                    st.metric("平均改进", f"{overall_improvement:.2f}%", delta_color="inverse")
                else:
                    st.warning("没有可用的MARD数据进行汇总")
            else:
                st.info("没有可用的MARD数据进行汇总")
    else:
        # 没有文件时的默认界面
        col1, col2 = st.columns([4, 1])

        with col1:
            if 'data' in st.session_state and not st.session_state.data.empty:
                fig = create_glucose_chart(st.session_state.data, "当前文件")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请上传Excel文件查看数据图表")

        if 'data' in st.session_state and not st.session_state.data.empty:
            display_data_preview(st.session_state.data, "当前文件")

    # ------------------------------
    # 应用说明
    # ------------------------------
    display_usage_instructions()


# ------------------------------
# 侧边栏配置函数
# ------------------------------

def configure_sidebar():
    """配置侧边栏控件"""
    # 模块选择器
    with st.sidebar.expander("🔧 选择处理模块", expanded=True):
        module_options = {
            'current_filter': '电流过滤模块',
            'decay_compensation': '衰减补偿模块',
            'depression_compensation': '凹陷补偿模块',
            'baseline_adjustment': '后期基线调整模块'
        }

        active_modules = st.multiselect(
            "选择要启用的处理模块:",
            options=list(module_options.keys()),
            format_func=lambda x: module_options[x],
            default=st.session_state.active_modules
        )

        if st.button("应用模块选择"):
            st.session_state.active_modules = active_modules
            st.experimental_rerun()

    # 公式选择器
    with st.sidebar.expander("📐 选择计算公式", expanded=True):
        global formula_options
        formula_options = {
            'formula1': '公式1: current * (a*slope² - b*slope + c) + d',
            'formula2': '公式2: current * ( slope * a + b) * c + d',
            'formula3': '公式3: (current + d) * (slope * a + b) * c',
            'formula4': '公式4: (current - intercept) / (slope / 100) / 18'
        }

        active_formulas = st.multiselect(
            "选择要使用的公式:",
            options=list(formula_options.keys()),
            format_func=lambda x: formula_options[x],
            default=st.session_state.active_formulas
        )

        if st.button("应用公式选择"):
            st.session_state.active_formulas = active_formulas
            st.experimental_rerun()

        # MARD偏移值设置
        st.session_state.shift_value = st.slider(
            "MARD计算偏移值:",
            min_value=0,
            max_value=50,
            value=st.session_state.shift_value,
            step=1,
            help="用于计算MARD的时间偏移"
        )

    # Y轴范围设置
    with st.sidebar.expander("📏 Y轴范围设置", expanded=True):
        st.subheader("电流值Y轴范围")
        current_min = st.number_input(
            "最小值",
            min_value=-50.0,
            max_value=50.0,
            value=float(st.session_state.yaxis_range['current_min']),
            step=0.1,
            format="%.1f"
        )
        current_max = st.number_input(
            "最大值",
            min_value=-50.0,
            max_value=100.0,
            value=float(st.session_state.yaxis_range['current_max']),
            step=0.1,
            format="%.1f"
        )

        st.subheader("血糖浓度Y轴范围")
        glucose_min = st.number_input(
            "最小值",
            min_value=0,
            max_value=100,
            value=st.session_state.yaxis_range['glucose_min'],
            step=1,
            format="%d"
        )
        glucose_max = st.number_input(
            "最大值",
            min_value=0,
            max_value=100,
            value=st.session_state.yaxis_range['glucose_max'],
            step=1,
            format="%d"
        )

        if st.button("应用Y轴范围"):
            st.session_state.yaxis_range = {
                'current_min': current_min,
                'current_max': current_max,
                'glucose_min': glucose_min,
                'glucose_max': glucose_max
            }
            st.experimental_rerun()

    # 图表配置
    with st.sidebar.expander("🎨 图表配置", expanded=True):
        configure_chart_settings()

    # 显示当前slope信息
    if st.session_state.single_slope is not None:
        st.sidebar.info(f"当前使用的单一slope值: **{st.session_state.single_slope:.6f}**")
        if st.session_state.slope_source:
            st.sidebar.caption(f"来源: {st.session_state.slope_source}")
        else:
            st.sidebar.caption("来源: 手动设置")
    else:
        st.sidebar.info("当前使用各数据点独立的slope值")

    # 显示列名映射信息
    if st.session_state.column_mapping:
        st.sidebar.subheader("列名映射")
        for key, value in st.session_state.column_mapping.items():
            st.sidebar.write(f"{key.upper()}: {value}")


def configure_chart_settings():
    """配置图表设置"""
    st.subheader("数据系列")

    # 电流系列配置
    st.markdown("**电流系列**")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state.chart_config['raw_current']['show'] = st.checkbox(
            "原始电流",
            value=st.session_state.chart_config['raw_current']['show'],
            key='show_raw_current'
        )

        st.session_state.chart_config['filtered_current']['show'] = st.checkbox(
            "过滤后电流",
            value=st.session_state.chart_config['filtered_current']['show'],
            key='show_filtered_current'
        )

    with col2:
        st.session_state.chart_config['decay_compensated_current']['show'] = st.checkbox(
            "衰减补偿电流",
            value=st.session_state.chart_config['decay_compensated_current']['show'],
            key='show_decay_current'
        )

        st.session_state.chart_config['depression_compensated_current']['show'] = st.checkbox(
            "凹陷补偿电流",
            value=st.session_state.chart_config['depression_compensated_current']['show'],
            key='show_depression_current'
        )

    # BGM点配置
    st.markdown("**参考血糖点**")
    st.session_state.chart_config['bgm_points']['show'] = st.checkbox(
        "显示参考血糖点",
        value=st.session_state.chart_config['bgm_points']['show'],
        key='show_bgm_points'
    )

    # 公式系列配置
    st.markdown("**公式系列**")
    st.session_state.chart_config['formula_original']['show'] = st.checkbox(
        "显示公式原始曲线",
        value=st.session_state.chart_config['formula_original']['show'],
        key='show_formula_original'
    )

    st.session_state.chart_config['formula_adjusted']['show'] = st.checkbox(
        "显示公式调整曲线",
        value=st.session_state.chart_config['formula_adjusted']['show'],
        key='show_formula_adjusted'
    )

    # 样式配置
    st.subheader("样式设置")

    # 确保宽度值为浮点数
    ensure_widths_are_float()

    # 线宽配置
    st.markdown("**线宽设置**")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state.chart_config['raw_current']['width'] = st.slider(
            "原始电流线宽",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.chart_config['raw_current']['width'],
            step=0.5,
            key='width_raw_current'
        )

        st.session_state.chart_config['filtered_current']['width'] = st.slider(
            "过滤电流线宽",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.chart_config['filtered_current']['width'],
            step=0.5,
            key='width_filtered_current'
        )

    with col2:
        st.session_state.chart_config['decay_compensated_current']['width'] = st.slider(
            "衰减电流线宽",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.chart_config['decay_compensated_current']['width'],
            step=0.5,
            key='width_decay_current'
        )

        st.session_state.chart_config['depression_compensated_current']['width'] = st.slider(
            "凹陷电流线宽",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.chart_config['depression_compensated_current']['width'],
            step=0.5,
            key='width_depression_current'
        )

    # 公式线宽
    st.session_state.chart_config['formula_original']['width'] = st.slider(
        "公式原始线宽",
        min_value=0.5,
        max_value=5.0,
        value=st.session_state.chart_config['formula_original']['width'],
        step=0.5,
        key='width_formula_original'
    )

    st.session_state.chart_config['formula_adjusted']['width'] = st.slider(
        "公式调整线宽",
        min_value=0.5,
        max_value=5.0,
        value=st.session_state.chart_config['formula_adjusted']['width'],
        step=0.5,
        key='width_formula_adjusted'
    )

    # 点大小
    st.session_state.chart_config['bgm_points']['width'] = st.slider(
        "参考点大小",
        min_value=2.0,
        max_value=15.0,
        value=st.session_state.chart_config['bgm_points']['width'],
        step=1.0,  # 确保为浮点数
        key='size_bgm_points'
    )

    # 线型配置
    st.markdown("**线型设置**")
    dash_options = {
        'solid': '实线',
        'dash': '虚线',
        'dot': '点线',
        'dashdot': '点划线'
    }

    st.session_state.chart_config['raw_current']['dash'] = st.selectbox(
        "原始电流线型",
        options=list(dash_options.keys()),
        format_func=lambda x: dash_options[x],
        index=list(dash_options.keys()).index(st.session_state.chart_config['raw_current']['dash']),
        key='dash_raw_current'
    )

    st.session_state.chart_config['filtered_current']['dash'] = st.selectbox(
        "过滤电流线型",
        options=list(dash_options.keys()),
        format_func=lambda x: dash_options[x],
        index=list(dash_options.keys()).index(st.session_state.chart_config['filtered_current']['dash']),
        key='dash_filtered_current'
    )

    st.session_state.chart_config['decay_compensated_current']['dash'] = st.selectbox(
        "衰减电流线型",
        options=list(dash_options.keys()),
        format_func=lambda x: dash_options[x],
        index=list(dash_options.keys()).index(st.session_state.chart_config['decay_compensated_current']['dash']),
        key='dash_decay_current'
    )

    st.session_state.chart_config['depression_compensated_current']['dash'] = st.selectbox(
        "凹陷电流线型",
        options=list(dash_options.keys()),
        format_func=lambda x: dash_options[x],
        index=list(dash_options.keys()).index(st.session_state.chart_config['depression_compensated_current']['dash']),
        key='dash_depression_current'
    )

    # 颜色配置
    st.subheader("颜色设置")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state.chart_config['raw_current']['color'] = st.color_picker(
            "原始电流颜色",
            value=st.session_state.chart_config['raw_current']['color'],
            key='color_raw_current'
        )

        st.session_state.chart_config['filtered_current']['color'] = st.color_picker(
            "过滤电流颜色",
            value=st.session_state.chart_config['filtered_current']['color'],
            key='color_filtered_current'
        )

        st.session_state.chart_config['decay_compensated_current']['color'] = st.color_picker(
            "衰减电流颜色",
            value=st.session_state.chart_config['decay_compensated_current']['color'],
            key='color_decay_current'
        )

        st.session_state.chart_config['depression_compensated_current']['color'] = st.color_picker(
            "凹陷电流颜色",
            value=st.session_state.chart_config['depression_compensated_current']['color'],
            key='color_depression_current'
        )

    with col2:
        st.session_state.chart_config['bgm_points']['color'] = st.color_picker(
            "参考点颜色",
            value=st.session_state.chart_config['bgm_points']['color'],
            key='color_bgm_points'
        )

    # 重置按钮
    if st.button("重置图表样式"):
        st.session_state.chart_config = {
            'raw_current': {'show': True, 'color': '#636EFA', 'width': 1.0, 'dash': 'solid'},
            'filtered_current': {'show': True, 'color': '#00B5F7', 'width': 1.0, 'dash': 'solid'},
            'decay_compensated_current': {'show': True, 'color': '#FFA15A', 'width': 1.0, 'dash': 'solid'},
            'depression_compensated_current': {'show': True, 'color': '#B82E2E', 'width': 1.0, 'dash': 'solid'},
            'bgm_points': {'show': True, 'color': '#FF6692', 'width': 8.0},
            'formula_original': {'show': True, 'width': 1.0, 'dash': 'dot'},
            'formula_adjusted': {'show': True, 'width': 3.0, 'dash': 'solid'},
        }
        st.experimental_rerun()


def ensure_widths_are_float():
    """确保所有宽度值为浮点数"""
    # 电流系列
    for key in ['raw_current', 'filtered_current', 'decay_compensated_current',
                'depression_compensated_current']:
        if isinstance(st.session_state.chart_config[key]['width'], list):
            st.session_state.chart_config[key]['width'] = float(st.session_state.chart_config[key]['width'][0])
        elif not isinstance(st.session_state.chart_config[key]['width'], float):
            st.session_state.chart_config[key]['width'] = float(st.session_state.chart_config[key]['width'])

    # 公式系列
    for key in ['formula_original', 'formula_adjusted']:
        if isinstance(st.session_state.chart_config[key]['width'], list):
            st.session_state.chart_config[key]['width'] = float(st.session_state.chart_config[key]['width'][0])
        elif not isinstance(st.session_state.chart_config[key]['width'], float):
            st.session_state.chart_config[key]['width'] = float(st.session_state.chart_config[key]['width'])

    # 点大小
    if isinstance(st.session_state.chart_config['bgm_points']['width'], list):
        st.session_state.chart_config['bgm_points']['width'] = float(
            st.session_state.chart_config['bgm_points']['width'][0])
    elif not isinstance(st.session_state.chart_config['bgm_points']['width'], float):
        st.session_state.chart_config['bgm_points']['width'] = float(
            st.session_state.chart_config['bgm_points']['width'])


def recalculate_glucose_for_data(data, file_name):
    """重新计算特定数据的血糖浓度"""
    try:
        # 获取该文件的参数
        file_params = st.session_state.file_params.get(file_name, st.session_state.params)

        # 确保所有输入都是数值类型
        if 'Depression Compensated Current' in data.columns:
            current_vals = pd.to_numeric(data['Depression Compensated Current'], errors='coerce')
        elif 'Decay Compensated Current' in data.columns:
            current_vals = pd.to_numeric(data['Decay Compensated Current'], errors='coerce')
        elif 'Filtered Current' in data.columns:
            current_vals = pd.to_numeric(data['Filtered Current'], errors='coerce')
        else:
            current_vals = pd.to_numeric(data['WE1 Current'], errors='coerce')

        slope_vals = pd.to_numeric(data['slope'], errors='coerce')

        # 计算所有激活公式的血糖浓度
        for formula in st.session_state.active_formulas:
            if formula == 'formula1':
                data[formula] = calculate_glucose1(
                    current_vals,
                    slope_vals,
                    file_params[formula]['a'],
                    file_params[formula]['b'],
                    file_params[formula]['c'],
                    file_params[formula]['d']
                )
            elif formula == 'formula2':
                data[formula] = calculate_glucose2(
                    current_vals,
                    slope_vals,
                    file_params[formula]['a'],
                    file_params[formula]['b'],
                    file_params[formula]['c'],
                    file_params[formula]['d']
                )
            elif formula == 'formula3':
                data[formula] = calculate_glucose3(
                    current_vals,
                    slope_vals,
                    file_params[formula]['a'],
                    file_params[formula]['b'],
                    file_params[formula]['c'],
                    file_params[formula]['d']
                )
            elif formula == 'formula4':
                data[formula] = calculate_glucose4(
                    current_vals,
                    slope_vals,
                    file_params[formula]['intercept'],
                )

            # 应用衰减基线调整
            if 'baseline_adjustment' in st.session_state.active_modules:
                data[f'{formula}_adjusted'] = compensate_stable_baseline(
                    data[formula].values
                )
            else:
                data[f'{formula}_adjusted'] = data[formula]

    except Exception as e:
        logger.error(f"血糖浓度重新计算失败: {str(e)}")


def display_mard_calculation(data, file_name):
    """显示单个文件的MARD计算，返回MARD结果"""
    if not data.empty and st.session_state.active_formulas:
        mard_results = []

        # 获取有效的BGM数据
        if 'BGM' in data.columns:
            valid_bgm = data['BGM'].dropna()
        else:
            valid_bgm = pd.Series(dtype=float)
            st.warning("数据中没有BGM列")

        if not valid_bgm.empty:
            for formula in st.session_state.active_formulas:
                if formula not in data.columns:
                    continue

                formula_name = formula_options[formula].split(':')[0]

                # 计算原始公式的MARD
                mard_orig, mad_orig = calculate_mard(
                    data[formula].values,
                    data['BGM'].values,
                    shift=st.session_state.shift_value
                )

                # 计算调整后公式的MARD
                if f'{formula}_adjusted' in data.columns:
                    mard_adj, mad_adj = calculate_mard(
                        data[f'{formula}_adjusted'].values,
                        data['BGM'].values,
                        shift=st.session_state.shift_value
                    )
                else:
                    mard_adj, mad_adj = np.nan, np.nan

                # 如果计算有效，添加到结果表
                if not np.isnan(mard_orig) and not np.isnan(mard_adj):
                    mard_results.append({
                        '文件': os.path.basename(file_name),
                        '公式': formula_name,
                        '原始MARD(%)': f"{mard_orig:.2f}",
                        '调整后MARD(%)': f"{mard_adj:.2f}",
                        '改进(%)': f"{(mard_orig - mard_adj):.2f}",
                        '偏移值': st.session_state.shift_value
                    })

            # 显示MARD结果表
            if mard_results:
                mard_df = pd.DataFrame(mard_results)
                st.dataframe(mard_df, hide_index=True)
            else:
                st.warning("无法计算MARD值：请检查偏移值设置")

            # 显示MARD说明
            st.caption(f"MARD值使用偏移值 {st.session_state.shift_value} 计算（数据点索引偏移）")
        else:
            st.warning("没有可用的BGM数据点计算MARD值")

        return mard_results
    else:
        return []


def display_data_preview(data, file_name):
    """显示数据预览和MARD计算"""
    if not data.empty and st.session_state.active_formulas:
        with st.expander("📊 数据预览与MARD计算", expanded=True):
            # 显示数据预览
            st.dataframe(data.head(10), height=300)

            # 计算并显示MARD值
            st.subheader("MARD值计算")

            # 获取有效的BGM数据
            if 'BGM' in data.columns:
                valid_bgm = data['BGM'].dropna()
            else:
                valid_bgm = pd.Series(dtype=float)
                st.warning("数据中没有BGM列")

            if not valid_bgm.empty:
                # 创建MARD结果表
                mard_results = []

                for formula in st.session_state.active_formulas:
                    if formula not in data.columns:
                        continue

                    formula_name = formula_options[formula].split(':')[0]

                    # 计算原始公式的MARD
                    mard_orig, mad_orig = calculate_mard(
                        data[formula].values,
                        data['BGM'].values,
                        shift=st.session_state.shift_value
                    )

                    # 计算调整后公式的MARD
                    if f'{formula}_adjusted' in data.columns:
                        mard_adj, mad_adj = calculate_mard(
                            data[f'{formula}_adjusted'].values,
                            data['BGM'].values,
                            shift=st.session_state.shift_value
                        )
                    else:
                        mard_adj, mad_adj = np.nan, np.nan

                    # 如果计算有效，添加到结果表
                    if not np.isnan(mard_orig) and not np.isnan(mard_adj):
                        mard_results.append({
                            '公式': formula_name,
                            '原始MARD(%)': f"{mard_orig:.2f}",
                            '调整后MARD(%)': f"{mard_adj:.2f}",
                            '改进(%)': f"{(mard_orig - mard_adj):.2f}",
                            '偏移值': st.session_state.shift_value
                        })

                # 显示MARD结果表
                if mard_results:
                    mard_df = pd.DataFrame(mard_results)
                    st.dataframe(mard_df, hide_index=True)
                else:
                    st.warning("无法计算MARD值：请检查偏移值设置")

                # 显示MARD说明
                st.caption(f"MARD值使用偏移值 {st.session_state.shift_value} 计算（数据点索引偏移）")
            else:
                st.warning("没有可用的BGM数据点计算MARD值")

            # 下载结果按钮
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载结果数据 (CSV)",
                data=csv,
                file_name='glucose_analysis_results.csv',
                mime='text/csv'
            )


def display_usage_instructions():
    """显示使用说明"""
    st.markdown("""
    ### 使用说明

    1. **上传数据**：
       - 支持多种列名格式：
         - 电流值列：`WE1 Current`, `电流值1`, `电流`, `current`等
         - 参考血糖列：`BGM`, `指尖血数据`, `参考值`, `参考血糖`等
         - Slope列：`Sen`, `Slope`, `斜率`, `sens`等
         - 序列号列：`Serial Number`, `Serial`, `序号`, `编号`等
       - 应用会自动检测匹配的列名

    2. **模块选择**：
       - 在侧边栏选择要启用的处理模块
       - 处理流程：电流过滤 → 衰减补偿 → 凹陷补偿 → 血糖计算 → 衰减基线调整
       - 每个模块独立运行，按顺序处理数据

    3. **公式选择**：
       - 选择要使用的计算公式
       - 支持同时选择多个公式进行对比
       - 每个公式有独立的参数设置区域

    4. **Y轴范围设置**：
       - 可自定义电流值和血糖浓度的Y轴范围
       - 默认电流值范围：-1 至 10 μA
       - 默认血糖浓度范围：0 至 20 mg/dL

    5. **数据可视化**：
       - **原始电流**：蓝色折线
       - **过滤后电流**：浅蓝色折线
       - **衰减补偿电流**：橙色折线
       - **凹陷补偿电流**：红色折线
       - **参考血糖值**：粉色点图
       - **计算血糖曲线**：每个公式生成两条曲线（原始和调整后）

    6. **MARD计算**：
       - 在数据预览区域显示每个公式的原始和调整后MARD值
       - 可调整偏移值（数据点索引偏移）
       - MARD值反映计算血糖曲线与参考血糖点的匹配程度
    """)


# ------------------------------
# 尝试导入filterpy库
# ------------------------------

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    st.warning("filterpy未安装，正在尝试从本地导入...")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'filterpy')))
    try:
        from filterpy.kalman import KalmanFilter
    except ImportError:
        st.error("无法加载filterpy库，电流过滤功能将不可用。请确保filterpy已安装或位于正确路径。")
        KalmanFilter = None


# ------------------------------
# 打包环境适配
# ------------------------------

def fix_streamlit_for_pyinstaller():
    """修复PyInstaller打包后Streamlit的资源路径问题"""
    import sys
    import os
    import streamlit.web.bootstrap as bootstrap

    if getattr(sys, 'frozen', False):
        # 如果是打包环境
        base_dir = sys._MEIPASS

        # 修复静态文件路径
        st_dir = os.path.join(base_dir, 'streamlit')
        bootstrap.STREAMLIT_STATIC_PATH = os.path.join(st_dir, 'static')

        # 修复配置文件路径
        os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

        # 创建必要的临时目录
        temp_dir = os.path.join(base_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['TEMP'] = temp_dir

        # 修复元数据路径
        if 'STREAMLIT_PACKAGE_METADATA' in os.environ:
            metadata_path = os.environ['STREAMLIT_PACKAGE_METADATA']
            if os.path.exists(metadata_path):
                # 创建符号链接
                link_path = os.path.join(st_dir, os.path.basename(metadata_path))
                if not os.path.exists(link_path):
                    try:
                        os.symlink(metadata_path, link_path)
                    except:
                        pass


# ------------------------------
# 运行应用
# ------------------------------

if __name__ == "__main__":
    # 修复打包环境问题
    fix_streamlit_for_pyinstaller()

    # 设置环境变量
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

    # 运行主函数
    main()