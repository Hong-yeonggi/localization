#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt


def read_pose_csv(csv_path):
    df = pd.read_csv(
        csv_path,
        sep=r'[,\s]+',
        engine='python',
        comment='#'
    )
    for col in ('x', 'y'):
        df[col] = df[col].astype(float)
    return df


def plot_single(csv_path, save_png=None):
    if not os.path.isfile(csv_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {csv_path}")
        return

    df = read_pose_csv(csv_path)

    fig, ax = plt.subplots()
    ax.plot(df['x'].to_numpy(), df['y'].to_numpy(),
            c='black', linestyle='--', label='GT')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # ★ 여기서 aspect equal 제거 ★
    # ax.set_aspect('equal', adjustable='datalim')

    # x축만 10~14로 고정 → 그래프 박스 크기는 그대로
    ax.set_xlim(-5, 15)

    ax.legend()
    ax.grid(True)

    if save_png:
        fig.savefig(save_png, dpi=150)
        print(f"[OK] 플롯 저장: {save_png}")
    else:
        plt.show()


def plot_multiple(csv_paths, save_png=None):
    fig, ax = plt.subplots()

    for p in csv_paths:
        if not os.path.isfile(p):
            print(f"[WARN] 파일 없음: {p}  → 건너뜀")
            continue
        df = read_pose_csv(p)
        ax.plot(df['x'].to_numpy(), df['y'].to_numpy(),
                linestyle='--', linewidth=1.0,
                label=os.path.basename(p))

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # ax.set_aspect('equal', adjustable='datalim')

    ax.set_xlim(0, 25)
    #ax.set_ylim(-5,4)
    ax.legend()
    ax.grid(True)

    if save_png:
        fig.savefig(save_png, dpi=150)
        print(f"[OK] 비교 플롯 저장: {save_png}")
    else:
        plt.show()


if __name__ == '__main__':
    single_csv = os.path.expanduser(
        '~/path/localization/amcl_logs/spot1/amcl_pose_spot1_base.csv')
    plot_single(
        single_csv,
        save_png=os.path.expanduser('~/path/localization/amcl_logs/spot1/base_spot1.png')
    )

    runs = [
        os.path.expanduser(
            '~/path/localization/amcl_logs/spot1/amcl_pose_spot1_base.csv'),
        os.path.expanduser(
            '~/path/localization/amcl_logs/spot1/amcl_pose_spot1_exp2.csv'),
        os.path.expanduser(
            '~/path/localization/amcl_logs/spot1/amcl_pose_spot1_score_exp1.csv')
    ]
    plot_multiple(
        runs,
        save_png=os.path.expanduser('~/path/localization/amcl_logs/spot1/compare_score.png')
    )
