#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from event_rules import detect_overspeeding, detect_harsh_braking, detect_idling
import plotly.express as px
from pathlib import Path

def load_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def preprocess(df):
    # Ensure numeric types
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    # Interpolate short gaps
    df['speed'] = df['speed'].interpolate(limit=5)
    df['latitude'] = df['latitude'].interpolate(limit=5)
    df['longitude'] = df['longitude'].interpolate(limit=5)
    # Smooth speed
    df['speed_smooth'] = df['speed'].rolling(window=3, min_periods=1).mean()
    # compute dt and acceleration (m/s^2 approx using km/h differences)
    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(1)
    # convert km/h to m/s for acceleration
    df['speed_mps'] = df['speed_smooth'] * (1000/3600)
    df['acceleration'] = df['speed_mps'].diff() / df['dt']
    return df

def visualize(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1 = px.line(df, x='timestamp', y='speed_smooth', title='Speed over Time')
    fig1.write_html(str(out_dir / 'speed_over_time.html'), auto_open=False)
    # Map visualization (requires internet for tiles if using mapbox)
    try:
        fig2 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='speed_smooth',
                                 zoom=12, height=600)
        fig2.update_layout(mapbox_style='open-street-map')
        fig2.write_html(str(out_dir / 'route_map.html'), auto_open=False)
    except Exception as e:
        print("Mapbox visualization skipped:", e)

def save_events(df, overspeed, harsh, idle, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overspeed.to_csv(out_dir / 'overspeed_events.csv', index=False)
    harsh.to_csv(out_dir / 'harsh_braking.csv', index=False)
    idle.to_csv(out_dir / 'idle_events.csv', index=False)
    summary = {
        'overspeed_count': len(overspeed),
        'harsh_braking_count': len(harsh),
        'idle_count': len(idle)
    }
    pd.Series(summary).to_json(out_dir / 'summary.json')

def main(input_path, out_dir):
    df = load_data(input_path)
    df = preprocess(df)
    overspeed = detect_overspeeding(df)
    harsh = detect_harsh_braking(df)
    idle = detect_idling(df)
    print(f"Overspeed events: {len(overspeed)}")
    print(f"Harsh braking events: {len(harsh)}")
    print(f"Idling events: {len(idle)}")
    save_events(df, overspeed, harsh, idle, out_dir)
    visualize(df, out_dir)
    print('Results saved to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Telematics Data Analyzer')
    parser.add_argument('--input', required=True, help='Path to telematics CSV file')
    parser.add_argument('--out', default='output', help='Output directory')
    args = parser.parse_args()
    main(args.input, args.out)
