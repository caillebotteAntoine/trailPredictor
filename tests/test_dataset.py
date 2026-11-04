# dataset.change_target("speed")
# dataset.prediction = dataset.target

# km_per_h = dataset.prediction.apply(lambda x: x.to("km_per_h"))
# print(km_per_h)
# activity_id = dataset._data["activity_id"]
# x = (
#     pd.DataFrame(
#         {"activity_id": activity_id, "speed": km_per_h, "target": dataset.target}
#     )
#     # .groupby("activity_id")["speed"]
#     .assign(time=pd.to_timedelta(dataset._segment_width / 1000 / km_per_h, unit="h"))
# )

# x["pace"] = [Speed.from_pace(p.total_seconds() / 60) for p in x["time"]]
# x
