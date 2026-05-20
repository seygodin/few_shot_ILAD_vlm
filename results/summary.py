import pandas as pd

df = pd.read_csv("train_hybrid_result.csv")

# 실제 데이터셋 이름 기준으로 표기 정규화(오타 대응)
df["data_name"] = (
    df["data_name"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)
df["data_name"] = df["data_name"].replace(
    {
        "blue_spliging": "blue_splicing",
        "blue_spiliging": "blue_splicing",
        "blue_spliging_": "blue_splicing",
        "red_spliging": "red_splicing",
        "yellow_spliging": "yellow_splicing",
    }
)

# 날짜 기준 분리를 위해 timestamp -> result_date 생성
df["result_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
df["result_date"] = df["result_date"].fillna("unknown_date")

group_cols = ["result_date", "data_name", "hybrid_model", "shot"]
metric_cols = [
    "best_la_auc",
    "hybrid_only_la_auc",
    "best_la_aupc",
    "hybrid_only_la_auprc",
    "best_la_f1",
    "hybrid_only_la_f1",
]

grouped = df.groupby(group_cols, as_index=False).agg(
    run_count=("best_la_auc", "count"),
    **{col: (col, "mean") for col in metric_cols},
)

# summary.csv에 실제로 기록되는 data_name 기준 정렬 순서
preferred_data_name_order = [
    "breakfast",
    "banana_juice",
    "orange_juice",
    "cherry_juice",
    "screw_bag",
    "blue_splicing",
    "red_splicing",
    "yellow_splicing",
    "pushpins",
    "capsule",
    "cable",
    "transistor",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
]
existing_names = grouped["data_name"].dropna().astype(str).unique().tolist()
data_name_order = [name for name in preferred_data_name_order if name in existing_names]
data_name_order += sorted([name for name in existing_names if name not in data_name_order])
grouped["data_name"] = pd.Categorical(
    grouped["data_name"],
    categories=data_name_order,
    ordered=True,
)
grouped = grouped.sort_values(
    by=["result_date", "data_name", "hybrid_model", "shot"],
    ascending=[True, True, True, True],
    na_position="last",
).reset_index(drop=True)

print(grouped)
grouped.to_csv("summary.csv", index=False)

# ---------------------------------------------------------
# 추가 집계:
# banana/orange/cherry_juice -> juice
# blue/red/yellow_splicing -> splicing
# 결과 data_name: breakfast, juice, screw_bag, splicing, pushpins (총 5개)
# ---------------------------------------------------------
merged_name_map = {
    "banana_juice": "juice",
    "orange_juice": "juice",
    "cherry_juice": "juice",
    "blue_splicing": "splicing",
    "red_splicing": "splicing",
    "yellow_splicing": "splicing",
}

df_5 = df.copy()
df_5["data_name"] = df_5["data_name"].replace(merged_name_map)

grouped_5 = df_5.groupby(group_cols, as_index=False).agg(
    run_count=("best_la_auc", "count"),
    **{col: (col, "mean") for col in metric_cols},
)

data_name_order_5 = ["breakfast", "juice", "screw_bag", "splicing", "pushpins", "capsule", "cable", "transistor", "pcb1", "pcb2", "pcb3", "pcb4"]
grouped_5["data_name"] = pd.Categorical(
    grouped_5["data_name"],
    categories=data_name_order_5,
    ordered=True,
)
grouped_5 = grouped_5.sort_values(
    by=["result_date", "data_name", "hybrid_model", "shot"],
    ascending=[True, True, True, True],
    na_position="last",
).reset_index(drop=True)

print(grouped_5)
grouped_5.to_csv("summary_5names.csv", index=False)
