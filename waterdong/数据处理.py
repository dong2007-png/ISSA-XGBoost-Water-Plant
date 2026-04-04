# -*- coding: utf-8 -*-
"""
整合 2021-2026 年原水和药耗数据（最终版 - 不填充缺失值）
- 只删除全空列，不填充任何缺失值
- 对数值列进行异常值裁剪（1%和99%分位数），忽略缺失值
- 保留原始数据模式（如周测指标的空缺）
"""

import os
import re
import pandas as pd
import numpy as np

# ================== 自动定位 data 文件夹 ==================
def find_data_folder(start_path):
    current = os.path.abspath(start_path)
    if os.path.isdir(os.path.join(current, "data")):
        return os.path.join(current, "data")
    parent = os.path.dirname(current)
    for _ in range(10):
        data_path = os.path.join(parent, "data")
        if os.path.isdir(data_path):
            return data_path
        if parent == os.path.dirname(parent):
            break
        parent = os.path.dirname(parent)
    return None

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = find_data_folder(script_dir)
if DATA_FOLDER is None:
    raise FileNotFoundError("未找到 data 文件夹")
OUTPUT_FOLDER = os.path.join(script_dir, "output_cleaned")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"数据文件夹: {DATA_FOLDER}")
print(f"输出文件夹: {OUTPUT_FOLDER}")

# ================== 特殊值转换函数 ==================
def convert_special_value(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ['', '/', '\\', '—', 'NaN', 'nan']:
        return np.nan
    if s.startswith('<'):
        return np.nan
    if s.startswith('>'):
        try:
            return float(s[1:])
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def clean_column_values(df):
    for col in df.columns:
        if col != '日期':
            df[col] = df[col].apply(convert_special_value)
    return df

# ================== 原水数据加载 ==================
def load_raw_water():
    all_data = []
    for file in os.listdir(DATA_FOLDER):
        if file.startswith("原水数据") and (file.endswith(".xls") or file.endswith(".xlsx")):
            print(f"读取原水文件: {file}")
            xls = pd.ExcelFile(os.path.join(DATA_FOLDER, file))
            for sheet in xls.sheet_names:
                df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
                if df_raw.empty:
                    continue
                has_time_row = False
                if len(df_raw) >= 2:
                    second_row = df_raw.iloc[1].astype(str).str.strip().tolist()
                    if any('点' in str(x) for x in second_row):
                        has_time_row = True
                if has_time_row:
                    col_names = df_raw.iloc[0].astype(str).str.strip().tolist()
                    time_row = df_raw.iloc[1].astype(str).str.strip().tolist()
                    new_cols = []
                    for i, name in enumerate(col_names):
                        if i < len(time_row) and time_row[i] not in ['nan', 'NaN', '']:
                            new_cols.append(f"{name}_{time_row[i]}")
                        else:
                            new_cols.append(name)
                    df = df_raw.iloc[3:].reset_index(drop=True)
                    df.columns = new_cols
                else:
                    col_names = df_raw.iloc[0].astype(str).str.strip().tolist()
                    df = df_raw.iloc[1:].reset_index(drop=True)
                    df.columns = col_names
                df = df.dropna(axis=1, how='all')
                date_col = None
                for col in df.columns:
                    if '日期' in col:
                        date_col = col
                        break
                if date_col is None:
                    print(f"  跳过 sheet {sheet}：未找到日期列")
                    continue
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                if df.empty:
                    continue
                df = df.rename(columns={date_col: '日期'})
                df = clean_column_values(df)
                if not df.empty:
                    all_data.append(df)
                    print(f"  成功读取 sheet {sheet}，{len(df)} 行，列数 {len(df.columns)}")
    if not all_data:
        raise Exception("未找到任何原水数据")
    raw = pd.concat(all_data, ignore_index=True)
    raw = raw.sort_values('日期').drop_duplicates('日期').reset_index(drop=True)
    print(f"原水数据合计 {len(raw)} 行，日期 {raw['日期'].min()} 至 {raw['日期'].max()}")
    return raw

# ================== 药耗数据加载 ==================
def load_dosage():
    all_data = []
    for file in os.listdir(DATA_FOLDER):
        if file.startswith("药耗数据") and (file.endswith(".xls") or file.endswith(".xlsx")):
            print(f"读取药耗文件: {file}")
            xls = pd.ExcelFile(os.path.join(DATA_FOLDER, file))
            for sheet in xls.sheet_names:
                df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
                if df_raw.empty:
                    continue
                if len(df_raw) < 3:
                    print(f"  跳过 sheet {sheet}：行数不足3")
                    continue
                col_names = df_raw.iloc[1].astype(str).str.strip().tolist()
                df = df_raw.iloc[2:].reset_index(drop=True)
                df.columns = col_names
                df = df.dropna(axis=1, how='all')
                date_col = None
                for col in df.columns:
                    if '日期' in col:
                        date_col = col
                        break
                if date_col is None:
                    print(f"  跳过 sheet {sheet}：未找到日期列")
                    continue
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                if df.empty:
                    continue
                df = df.rename(columns={date_col: '日期'})
                df = clean_column_values(df)
                target_col = None
                for col in df.columns:
                    if '耗用矾量' in col or ('矾' in col and 'kg' in col):
                        target_col = col
                        break
                if target_col is not None:
                    df = df[df[target_col] > 0]
                if not df.empty:
                    all_data.append(df)
                    print(f"  成功读取 sheet {sheet}，{len(df)} 行，列数 {len(df.columns)}")
    if not all_data:
        raise Exception("未找到任何药耗数据")
    dosage = pd.concat(all_data, ignore_index=True)
    dosage = dosage.sort_values('日期').drop_duplicates('日期').reset_index(drop=True)
    print(f"药耗数据合计 {len(dosage)} 行，日期 {dosage['日期'].min()} 至 {dosage['日期'].max()}")
    return dosage

# ================== 数据清洗（只裁剪异常值，不填充缺失值） ==================
def clean_data(df):
    """只删除全空列，对数值列进行异常值裁剪（1%和99%分位数），不填充缺失值"""
    df = df.copy()
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().all():
            df = df.drop(columns=[col])
            print(f"  删除全空列: {col}")
        else:
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            if q1 != q3:
                df[col] = df[col].clip(q1, q3)
                print(f"  裁剪列 {col} 异常值（1%-99%分位数）")
    return df

def save_both(df, name):
    csv_path = os.path.join(OUTPUT_FOLDER, f"{name}.csv")
    excel_path = os.path.join(OUTPUT_FOLDER, f"{name}.xlsx")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"已保存: {csv_path} 和 {excel_path}")

def main():
    print("开始整合数据...")
    raw = load_raw_water()
    dosage = load_dosage()
    print("清洗数据（只裁剪异常值，不填充缺失值）...")
    raw_clean = clean_data(raw)
    dosage_clean = clean_data(dosage)
    merged = pd.merge(raw_clean, dosage_clean, on='日期', how='inner')
    print(f"合并后共 {len(merged)} 行，日期范围 {merged['日期'].min()} 至 {merged['日期'].max()}")
    save_both(raw_clean, "cleaned_raw_water")
    save_both(dosage_clean, "cleaned_dosage")
    save_both(merged, "merged_cleaned_data")
    print("全部完成！")

if __name__ == "__main__":
    main()