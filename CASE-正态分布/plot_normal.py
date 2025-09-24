#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制给定均值与标准差的正态分布曲线，并保存为PNG图片。

默认参数：
- 均值(mean) = 179.5
- 标准差(std_dev) = 3.697

使用说明：
1) 直接运行：python plot_normal.py
2) 自定义参数：python plot_normal.py --mean 170 --std 5

注意：
- 本脚本尽量避免依赖 SciPy，仅用 NumPy/Matplotlib。
- 中文显示：尝试使用系统中常见中文字体；若无，仍可运行但中文可能fallback。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class NormalParams:
    """正态分布参数容器。"""
    mean: float
    std_dev: float


def setup_chinese_font() -> None:
    """尽量设置中文字体，避免中文乱码与负号显示问题。

    - 优先尝试 "SimHei"（黑体），Windows 常见。
    - 回退为 Matplotlib 默认字体。
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Noto Sans CJK SC']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    except Exception:
        # 安全降级：不抛出异常，保持默认
        pass


def compute_normal_pdf(x: np.ndarray, params: NormalParams) -> np.ndarray:
    """计算正态分布概率密度函数值。

    使用标准正态分布公式：
    pdf(x) = 1/(σ*√(2π)) * exp(-(x-μ)^2 / (2σ^2))
    """
    mean = params.mean
    std = params.std_dev
    if std <= 0:
        raise ValueError("标准差必须为正数")

    coef = 1.0 / (std * math.sqrt(2.0 * math.pi))
    return coef * np.exp(-((x - mean) ** 2) / (2.0 * std * std))


def make_x_range(params: NormalParams, num_points: int = 1000, width_sigma: float = 4.0) -> np.ndarray:
    """生成绘图用 x 轴范围：以均值为中心，覆盖 ±width_sigma 个标准差。"""
    half_width = width_sigma * params.std_dev
    x_min = params.mean - half_width
    x_max = params.mean + half_width
    # 保证范围有效
    if not np.isfinite([x_min, x_max]).all() or x_min >= x_max:
        raise ValueError("生成的 x 轴范围不合法，请检查参数")
    return np.linspace(x_min, x_max, num_points)


def create_plot(params: NormalParams) -> Tuple[plt.Figure, plt.Axes]:
    """绘制正态分布曲线并返回 Figure 与 Axes。"""
    setup_chinese_font()

    x = make_x_range(params)
    y = compute_normal_pdf(x, params)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, color="#2E86DE", linewidth=2.0, label="概率密度函数")

    # 均值与±1σ标记
    mean = params.mean
    std = params.std_dev
    ax.axvline(mean, color="#222222", linestyle="--", linewidth=1.2, label=f"均值 μ={mean:.3f}")
    ax.axvline(mean - std, color="#E67E22", linestyle=":", linewidth=1.2, label=f"μ-σ={mean-std:.3f}")
    ax.axvline(mean + std, color="#27AE60", linestyle=":", linewidth=1.2, label=f"μ+σ={mean+std:.3f}")

    # 填充 ±1σ 范围面积（约占 68.27%）
    x_fill = np.linspace(mean - std, mean + std, 400)
    y_fill = compute_normal_pdf(x_fill, params)
    ax.fill_between(x_fill, y_fill, color="#27AE601A")  # 透明度通过十六进制后两位设置

    ax.set_title("正态分布（Normal Distribution）", fontsize=14)
    ax.set_xlabel("随机变量 x", fontsize=12)
    ax.set_ylabel("概率密度 f(x)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()

    return fig, ax


def save_figure(fig: plt.Figure, filename: str) -> str:
    """保存图像到指定文件名，返回保存的文件路径。"""
    fig.savefig(filename, dpi=160)
    return filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制正态分布曲线并保存图片。")
    parser.add_argument("--mean", type=float, default=179.5, help="均值 μ，默认 179.5")
    parser.add_argument("--std", type=float, default=3.697, help="标准差 σ，默认 3.697")
    parser.add_argument("--output", type=str, default="normal_179_5_3_697.png", help="输出PNG文件名")
    parser.add_argument("--show", action="store_true", help="绘制后显示窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 参数与容错
    try:
        params = NormalParams(mean=float(args.mean), std_dev=float(args.std))
        if not np.isfinite([params.mean, params.std_dev]).all():
            raise ValueError("均值或标准差不是有限数值")
        if params.std_dev <= 0:
            raise ValueError("标准差必须为正数")
    except Exception as exc:
        print(f"参数错误：{exc}")
        return

    # 绘图
    try:
        fig, _ = create_plot(params)
    except Exception as exc:
        print(f"绘图失败：{exc}")
        return

    # 保存
    try:
        # 若用户未自定义输出名且参数为默认值，则使用默认文件名
        default_name = "normal_179_5_3_697.png"
        filename = args.output if args.output else default_name
        saved = save_figure(fig, filename)
        print(f"图片已保存：{saved}")
    except Exception as exc:
        print(f"保存图片失败：{exc}")
    finally:
        if args.show:
            try:
                plt.show()
            except Exception:
                # 在无GUI环境下可能报错，忽略显示
                pass
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()


