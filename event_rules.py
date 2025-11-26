import pandas as pd
def detect_overspeeding(df, limit=60):
    return df[df['speed_smooth'] > limit]

def detect_harsh_braking(df, decel_threshold=-3.5):
    # acceleration in m/s^2, detect where acc < threshold
    if 'acceleration' not in df.columns:
        return df.iloc[0:0]
    return df[df['acceleration'] < decel_threshold]

def detect_idling(df, idle_time_sec=120):
    # identify consecutive rows where speed is (near) zero
    df = df.copy()
    df['is_idle'] = (df['speed_smooth'] <= 0.5).astype(int)
    # compute consecutive idle durations in seconds
    df['idle_group'] = (df['is_idle'] != df['is_idle'].shift()).cumsum()
    result = []
    for _, g in df.groupby('idle_group'):
        if g['is_idle'].iloc[0] == 1:
            duration = g['timestamp'].iloc[-1] - g['timestamp'].iloc[0]
            if duration.total_seconds() >= idle_time_sec:
                result.append(g)
    if result:
        return pd.concat(result)
    return df.iloc[0:0]
