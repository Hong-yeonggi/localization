import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 우리가 쓸 표준 열 이름
COLS = ['t_sec', 'x', 'y', 'yaw_rad']

def read_pose(path: str | Path) -> pd.DataFrame:
    """
    AMCL pose 로그를 헤더/구분자 종류와 무관하게 DataFrame으로 반환.
    • 콤마 CSV 또는 공백/탭 구분 파일 모두 지원
    • 헤더 없으면 자동으로 이름을 부여
    • 줄 끝 CR(^M) 문자 제거
    """
    path = Path(path)
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()

    # ── ① 구분자 및 헤더 유무 판별 ─────────────────────────────
    comma_sep   = ',' in first          # 콤마가 있으면 CSV
    has_header  = first.lower().startswith('t_sec')

    read_cfg = dict(
        sep        = ',' if comma_sep else r'\s+',   # ',' 또는 모든 공백
        engine     = 'python',                       # 정규식 구분자 허용
        comment    = '#',                            # 주석 행 무시
        header     = 0 if has_header else None,
        names      = None if has_header else COLS,
    )

    df = pd.read_csv(path, **read_cfg)

    # ── ② 헤더에 붙은 CR 문자 제거 후 컬럼 선택 ──────────────
    df.columns = df.columns.str.strip()              # 'yaw_rad\r' → 'yaw_rad'
    if not set(COLS).issubset(df.columns):           # 헤더가 이상하면 names로 재호출
        df = pd.read_csv(path, sep=read_cfg['sep'], engine='python',
                         header=None, names=COLS, comment='#')
    df = df[COLS].astype(float)                      # 문자열 → float

    return df

# === 사용 예시 ===
file1 = '/home/csilab/tb3/path/amcl_skip_spot12.csv'
file2 = '/home/csilab/tb3/path/amcl_base.csv'

df1 = read_pose(file1).sort_values('t_sec')
df2 = read_pose(file2).sort_values('t_sec')

# === 플롯 ===
fig, ax = plt.subplots()

ax.plot(df1['x'].to_numpy(), df1['y'].to_numpy(),
        '-', lw=1.5, label='amcl_skip_spot12')
ax.plot(df2['x'].to_numpy(), df2['y'].to_numpy(),
        '-', lw=1.5, label='amcl_base')

# ▷ 여기서 축 범위를 직접 지정하세요 ◁
ax.set_xlim(12, 14)     # xmin, xmax
ax.set_ylim(-35, 0)     # ymin, ymax

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('AMCL trajectory comparison')
ax.set_aspect('equal', adjustable='box')
ax.legend()
plt.tight_layout()
plt.show()
