# -*- coding: utf-8 -*-
"""
fetch_real_data.py
==================
Fetch REAL data for the Common Prosperity Index (CPI) project from the
National Bureau of Statistics of China (国家统计局) via akshare, and write
data/common_prosperity_data.csv in the SAME schema (header order + 31 provinces
+ 8 indicators) that the existing CNN/LSTM pipeline expects.

Data source
-----------
akshare -> 国家统计局 分省年度数据 (kind='分省年度数据', code=6, route='fsYearData')
Interface: ak.macro_china_nbs_region(kind, path, indicator, region=None, period)

Indicator mapping (real official 口径)
--------------------------------------
Column                         Path (目录)                                             Official indicator (口径)
GDP_per_capita                 国民经济核算 > 地区生产总值                            人均地区生产总值 (元/人)
Disposable_income              人民生活 > 居民人均可支配收入                          全体居民人均可支配收入 (元)
Urbanization_rate              人口 > 总人口                                          城镇人口 / 年末常住人口  (derived ratio)
Education_ratio                教育 > 每十万人口各级教育平均在校生数                  每十万人口高等教育平均在校生数 (人)
Social_security                公共管理、社会保障及其他 > 城镇职工基本养老保险        参加城镇职工基本养老保险人数 / 年末常住人口 (derived coverage)
Urban_rural_income_ratio       人民生活 > 居民人均可支配收入                          城镇居民人均可支配收入 ÷ 农村居民人均可支配收入 (derived ratio)
Environment_index              城市概况 > 城市绿地和园林                              建成区绿化覆盖率 (%)  [positive: 越高越好]
Infrastructure_index           运输和邮电 > 互联网主要指标发展情况                      互联网宽带接入用户 ÷ 年末常住人口 (derived per-capita proxy)

Substitution notes (替代关系，如实说明):
  * Urbanization_rate: 统计局"分省年度"无单列"城镇化率"指标，改用
    「城镇人口 ÷ 年末常住人口」的比例（同一"人口 > 总人口"目录下的两个真实指标）。
  * Social_security: 用「参加城镇职工基本养老保险人数 ÷ 年末常住人口」作为
    社保覆盖率的代理（城镇职工口径，反映正规部门社保覆盖水平）。原始值本身
    是"人数(万人)"，除以常住人口得到覆盖率以避免人口规模带来的偏差。
  * Environment_index: 原"单位地区生产总值能耗(等价值)"在统计局分省年度接口中
    **返回空值列（该指标分省数据不可用）**，故改用「城市概况 > 城市绿地和园林 >
    建成区绿化覆盖率(%)」——真实、2013-2023 全覆盖、正向指标(越高环境越好)。
  * Infrastructure_index: 直接用「邮电业务总量」时其分省数据仅覆盖 2013-2017
    (缺 2018-2023)，故改用覆盖 2013-2023 全年的「互联网宽带接入用户(万户) ÷
    年末常住人口(万人)」= 宽带渗透率，避免总量受人口/面积规模影响。
  * 所有"÷"均为把官方绝对量指标转换为可跨省对比的人均/比例口径，分子分母
    都来自国家统计局真实原始数据，未编造任何数值。

Normalization & CPI
-------------------
- 正向指标: min-max 正向。(x-min)/(max-min)
- 负向指标: 仅 Urban_rural_income_ratio (城乡收入比越小越好) 做反向
  (max-x)/(max-min)。Environment_index(建成区绿化覆盖率)为正向。
- CPI 采用**熵值法(熵权法)**：
    p_ij = norm_ij / Σ_i norm_ij                       (指标 j 下各省份-年份占比)
    e_j  = -k Σ_i p_ij ln p_ij ,  k = 1/ln(n)          (信息熵, 0·ln0=0)
    d_j  = 1 - e_j                                     (差异系数)
    w_j  = d_j / Σ_j d_j                               (权重)
    CPI_i = Σ_j w_j · norm_ij                          (综合得分)
  注意: 归一化与权重在"省份×年份"的合并面板上统一计算(与项目原有合成脚本
  的全局 min-max 口径保持一致)。

Missing values: 按省份对各指标时间序列做线性插值(x.interpolate) + 前向/后向填充。

Offline behaviour: 若所有抓取均失败, 抛 RuntimeError, 不产出空文件。
"""

import os
import sys
import time
import shutil

import numpy as np
import pandas as pd
import akshare as ak
import akshare.economic.macro_china_nbs as M

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KIND = '分省年度数据'
PERIOD = '2013-2023'                     # 目标区间；实际拿到多少年就多少年
DATA_DIR = 'data'
OUT_CSV = os.path.join(DATA_DIR, 'common_prosperity_data.csv')
BACKUP_CSV = os.path.join(DATA_DIR, 'common_prosperity_data_synthetic_backup.csv')

# 31 provinces, English names EXACTLY as in generate_data.py
PROVINCES = [
    'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'InnerMongolia', 'Liaoning', 'Jilin', 'Heilongjiang',
    'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong', 'Henan',
    'Hubei', 'Hunan', 'Guangdong', 'Guangxi', 'Hainan', 'Chongqing', 'Sichuan', 'Guizhou',
    'Yunnan', 'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
]

# Chinese (NBS) -> English mapping
CN2EN = {
    '北京': 'Beijing', '天津': 'Tianjin', '河北': 'Hebei', '山西': 'Shanxi',
    '内蒙古': 'InnerMongolia', '辽宁': 'Liaoning', '吉林': 'Jilin', '黑龙江': 'Heilongjiang',
    '上海': 'Shanghai', '江苏': 'Jiangsu', '浙江': 'Zhejiang', '安徽': 'Anhui',
    '福建': 'Fujian', '江西': 'Jiangxi', '山东': 'Shandong', '河南': 'Henan',
    '湖北': 'Hubei', '湖南': 'Hunan', '广东': 'Guangdong', '广西': 'Guangxi',
    '海南': 'Hainan', '重庆': 'Chongqing', '四川': 'Sichuan', '贵州': 'Guizhou',
    '云南': 'Yunnan', '西藏': 'Tibet', '陕西': 'Shaanxi', '甘肃': 'Gansu',
    '青海': 'Qinghai', '宁夏': 'Ningxia', '新疆': 'Xinjiang',
}

POSITIVE = ['GDP_per_capita', 'Disposable_income', 'Urbanization_rate',
            'Education_ratio', 'Social_security', 'Environment_index',
            'Infrastructure_index']
# 负向指标（越小越好）：城乡收入比
NEGATIVE = ['Urban_rural_income_ratio']

INDICATORS = ['GDP_per_capita', 'Disposable_income', 'Urbanization_rate',
              'Education_ratio', 'Social_security', 'Urban_rural_income_ratio',
              'Environment_index', 'Infrastructure_index']

# exact CSV header (must match existing file order)
HEADER = ['Province', 'Year',
          'GDP_per_capita', 'Disposable_income', 'Urbanization_rate', 'Education_ratio',
          'Social_security', 'Urban_rural_income_ratio', 'Environment_index',
          'Infrastructure_index',
          'GDP_per_capita_norm', 'Disposable_income_norm', 'Urbanization_rate_norm',
          'Education_ratio_norm', 'Social_security_norm', 'Environment_index_norm',
          'Infrastructure_index_norm', 'Urban_rural_income_ratio_norm', 'CPI']

FETCH_LOG = {}   # indicator key -> status string


# ---------------------------------------------------------------------------
# Robust NBS fetch helpers (handle rate-limit / transient JS challenge pages)
# ---------------------------------------------------------------------------
def clear_caches():
    """Purge akshare's @lru_cache so a transient empty response can't be cached."""
    for fn in (M._get_nbs_tree, M._get_nbs_root_id, M._get_nbs_indicators,
               M._get_nbs_da_catalogs, M._get_nbs_da_members):
        try:
            fn.cache_clear()
        except Exception:
            pass


def fetch(path, indicator, period=PERIOD, tries=8):
    """
    Fetch one indicator (index=省份, columns=年份) with retries.
    Returns DataFrame or None.
    """
    last_err = None
    for i in range(tries):
        clear_caches()
        try:
            df = ak.macro_china_nbs_region(
                kind=KIND, path=path, indicator=indicator,
                region=None, period=period)
            if df is not None and len(df) > 0:
                return df
            last_err = 'empty result'
        except Exception as e:
            last_err = repr(e)[:160]
        time.sleep(1.5 + 1.5 * i)
    print(f"    [!] fetch failed ({indicator!r} @ {path!r}): {last_err}")
    return None


def norm_cn(s):
    """Normalize a Chinese province name (strip 省/市/自治区 suffixes)."""
    s = str(s).strip()
    for suf in ('维吾尔自治区', '回族自治区', '壮族自治区', '自治区',
                '特别行政区', '省', '市'):
        if s.endswith(suf):
            return s[:-len(suf)]
    return s


def cn_to_en(name):
    return CN2EN.get(norm_cn(name))


def to_long(df, value_name):
    """Wide (index=region, cols=year) -> long with columns [Province, Year, value_name]."""
    if df is None:
        return None
    recs = []
    for region in df.index:
        en = cn_to_en(region)
        if en is None:
            continue
        for col in df.columns:
            ys = ''.join(ch for ch in str(col) if ch.isdigit())
            if len(ys) != 4:
                continue
            val = df.loc[region, col]
            if pd.isna(val):
                continue
            recs.append({'Province': en, 'Year': int(ys), value_name: float(val)})
    if not recs:
        return None
    return pd.DataFrame(recs)


def merge_on(base, extra, key):
    if extra is None:
        return None
    return base.merge(extra, on=['Province', 'Year'], how='outer') if base is not None \
        else extra


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 74)
    print("  Fetching REAL data from 国家统计局 (分省年度数据, 2013-2023)")
    print("=" * 74)

    def grab(label, path, indicator):
        print(f"  - {label}: {path} > {indicator}")
        df = fetch(path, indicator)
        FETCH_LOG[label] = 'OK' if df is not None else 'FAIL'
        if df is None:
            return None
        df = to_long(df, label)
        if df is None:
            FETCH_LOG[label] = 'FAIL(no-rows)'
        return df

    # --- 1. GDP per capita ---
    gdp = grab('GDP_per_capita', '国民经济核算 > 地区生产总值',
               '人均地区生产总值 (元/人) ')

    # --- 2. Disposable income (全体居民) ---
    inc_all = grab('Disposable_income', '人民生活 > 居民人均可支配收入',
                   '全体居民人均可支配收入 (元) ')
    inc_urban = grab('_inc_urban', '人民生活 > 居民人均可支配收入',
                     '城镇居民人均可支配收入 (元) ')
    inc_rural = grab('_inc_rural', '人民生活 > 居民人均可支配收入',
                     '农村居民人均可支配收入 (元) ')

    # --- 3. Population (total & urban) ---
    pop_total = grab('_pop_total', '人口 > 总人口', '年末常住人口 (万人) ')
    pop_urban = grab('_pop_urban', '人口 > 总人口', '城镇人口 (万人) ')

    # --- 4. Education ---
    edu = grab('Education_ratio', '教育 > 每十万人口各级教育平均在校生数',
               '每十万人口高等教育平均在校生数 (人) ')

    # --- 5. Social security (城镇职工养老保险参保人数) ---
    sec = grab('_sec_insured', '公共管理、社会保障及其他 > 城镇职工基本养老保险',
               '参加城镇职工基本养老保险人数 (万人) ')

    # --- 6. Environment (建成区绿化覆盖率, 正向) ---
    env = grab('Environment_index', '城市概况 > 城市绿地和园林',
               '建成区绿化覆盖率 (%)')

    # --- 7. Infrastructure (互联网宽带接入用户, 之后除以人口 -> 渗透率) ---
    infra = grab('_broadband', '运输和邮电 > 互联网主要指标发展情况',
                 '互联网宽带接入用户 (万户) ')

    # ---- assemble ----
    base = gdp if gdp is not None else inc_all
    if base is None:
        raise RuntimeError(
            "网络抓取全部失败：无法连接国家统计局接口。"
            "请检查网络后重试；脚本不会产出一个空/假的数据文件。")

    df = base.copy()
    for extra in (inc_all, inc_urban, inc_rural, pop_total, pop_urban,
                  edu, sec, env, infra):
        if extra is not None:
            df = df.merge(extra, on=['Province', 'Year'], how='outer')

    # derived indicators
    if '_inc_urban' in df and '_inc_rural' in df:
        df['Urban_rural_income_ratio'] = df['_inc_urban'] / df['_inc_rural']
        FETCH_LOG['Urban_rural_income_ratio'] = 'OK (derived: 城镇/农村 可支配收入)'

    if '_pop_urban' in df and '_pop_total' in df:
        df['Urbanization_rate'] = df['_pop_urban'] / df['_pop_total']
        FETCH_LOG['Urbanization_rate'] = 'OK (derived: 城镇人口/常住人口)'

    if '_sec_insured' in df and '_pop_total' in df:
        df['Social_security'] = df['_sec_insured'] / df['_pop_total']
        FETCH_LOG['Social_security'] = 'OK (derived: 职工养老参保/常住人口)'

    if '_broadband' in df and '_pop_total' in df:
        df['Infrastructure_index'] = df['_broadband'] / df['_pop_total']
        FETCH_LOG['Infrastructure_index'] = 'OK (derived: 宽带接入用户/常住人口)'

    # ---- keep 31 provinces, real indicators only ----
    df = df[df['Province'].isin(PROVINCES)].copy()

    have = [c for c in INDICATORS if c in df.columns]
    missing_cols = [c for c in INDICATORS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"以下指标整列缺失，无法生成: {missing_cols}. "
            f"抓取日志: {FETCH_LOG}")

    df = df[['Province', 'Year'] + INDICATORS].copy()
    df = df.sort_values(['Province', 'Year']).reset_index(drop=True)

    # ---- missing value handling: per-province linear interpolation + ffill/bfill ----
    filled_report = {}
    for col in INDICATORS:
        before = int(df[col].isna().sum())
        df[col] = df.groupby('Province')[col].transform(
            lambda s: s.interpolate(method='linear', limit_direction='both'))
        df[col] = df.groupby('Province')[col].transform(
            lambda s: s.ffill().bfill())
        after = int(df[col].isna().sum())
        filled_report[col] = (before, after)
        if after > 0:
            raise RuntimeError(f"指标 {col} 省份内仍有缺失无法填补: {after} 行")

    # drop rows where a province has no year coverage at all
    df = df.dropna(subset=INDICATORS).reset_index(drop=True)

    # ---- normalization (global min-max over province-year panel) ----
    for col in POSITIVE:
        mn, mx = df[col].min(), df[col].max()
        df[col + '_norm'] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0
    for col in NEGATIVE:
        mn, mx = df[col].min(), df[col].max()
        df[col + '_norm'] = (mx - df[col]) / (mx - mn) if mx > mn else 0.0

    norm_cols = [c + '_norm' for c in INDICATORS]

    # ---- CPI via entropy weight method (熵值法) ----
    # p_ij = norm_ij / sum_i norm_ij ; e_j = -k sum p_ij ln p_ij ; d_j = 1-e_j ; w_j = d_j/sum d
    X = df[norm_cols].values.astype(float)
    n = X.shape[0]
    col_sum = X.sum(axis=0)
    col_sum[col_sum == 0] = 1e-12
    P = X / col_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    k = 1.0 / np.log(n)
    e = -k * plogp.sum(axis=0)
    d = 1.0 - e
    d = np.where(d < 0, 0.0, d)          # 数值安全
    w = d / d.sum() if d.sum() > 0 else np.ones_like(d) / len(d)
    df['CPI'] = (X * w.reshape(1, -1)).sum(axis=1)

    # ---- write, with backup ----
    if os.path.exists(OUT_CSV) and not os.path.exists(BACKUP_CSV):
        shutil.copyfile(OUT_CSV, BACKUP_CSV)
        print(f"  [backup] {OUT_CSV} -> {BACKUP_CSV}")

    df = df[HEADER]
    df.to_csv(OUT_CSV, index=False, encoding='utf-8')

    # ---- report ----
    print("-" * 74)
    print(f"  Saved: {OUT_CSV}   rows={len(df)}  provinces={df['Province'].nunique()}  "
          f"years={int(df['Year'].min())}-{int(df['Year'].max())}")
    print("  Fetch status:")
    for k_, v_ in FETCH_LOG.items():
        print(f"    {k_:<26} {v_}")
    print("  Missing-fill (before -> after, per province interpolation):")
    for col in INDICATORS:
        b, a = filled_report[col]
        print(f"    {col:<26} {b} -> {a}")
    print("  Entropy weights (熵值法):")
    for c, wi in zip(INDICATORS, w):
        print(f"    {c:<26} w={wi:.4f}")
    print("  CPI range: [%.4f, %.4f]  mean=%.4f" % (df['CPI'].min(), df['CPI'].max(), df['CPI'].mean()))
    print("=" * 74)

    return df


if __name__ == '__main__':
    main()
