import pandas as pd
import numpy as np

df = pd.read_csv("input_final_병합.csv")

df['beam_idx_val'] = df['beam_idx'] 
    id_vars=['scan_time', 'beam_idx'],
    value_vars=['beam_idx_val', 'range', 'intensity', 'score'],
    var_name='feature',
    value_name='val'
)
df_wide = df_melt.pivot_table(
    index='scan_time',
    columns=['beam_idx', 'feature'],
    values='val'
)

def flatten(cols):
    return [f"{int(b)}_{feat}" for b, feat in cols]
df_wide.columns = flatten(df_wide.columns)

all_cols   = df_wide.columns
score_cols = [c for c in all_cols if c.endswith("_score")]
input_cols = [c for c in all_cols if c.endswith("_range") or c.endswith("_intensity")]

df_input = df_wide[input_cols].copy()
df_score = df_wide[score_cols].copy()

# 4) 0~359 빔까지 없는 컬럼 채우기
max_beam = 360
for b in range(max_beam):
    r = f"{b}_range"
    i = f"{b}_intensity"
    if r not in df_input.columns:
        df_input[r] = np.nan
    if i not in df_input.columns:
        df_input[i] = np.nan

df_input.fillna(0, inplace=True)
df_score.fillna(0, inplace=True)

final_cols = []
for b in range(max_beam):
    final_cols += [f"{b}_range", f"{b}_intensity"]
df_input_final = df_input[final_cols]

df_input_final.to_csv("input_2N_final.csv", index=True)
df_score.to_csv("score_2N_final.csv", index=True)

print("Saved ")
