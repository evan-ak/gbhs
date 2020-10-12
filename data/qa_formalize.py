
import os
import json

local_path = os.path.abspath(".")
_sets = ["train1", "validation1", "validation2", "test1", "test2"]
for _set in _sets :
  file_in = local_path + "/" + _set + "/qa_pairs.json"
  file_type = local_path + "/" + _set + "/types.json"
  file_out = local_path + "/" + _set + "/questions_grouped.json"
  # file_idx = local_path + "/" + _set + "/questions_indexes.json"

  with open(file_in, "r") as f :
    data_in = json.load(f)
  with open(file_type, "r") as f :
    types = json.load(f)
  data_out = {}
  # data_idx = {}

  for idx, log in enumerate(data_in["qa_pairs"]) :
    data = []
    data.append(log["question_id"])
    data.append(types[log["image_index"]])
    data.append(log["image_index"])
    data.append(log["question_string"])
    if "answer" in log :
      data.append(log["answer"])
    else :
      data.append(None)
    if "None" in log["color1_name"] :
      data.append(None)
    else :
      data.append(log["color1_name"])
      data[3] = data[3].replace(data[-1], "OBJ1")
    if "None" in log["color2_name"] :
      data.append(None)
    else :
      data.append(log["color2_name"])
      data[3] = data[3].replace(data[-1], "OBJ2")
    data.append(idx)

    if not data[0] in data_out :
      data_out[data[0]] = {}
      # data_idx[data[0]] = {}
    if not data[1] in data_out[data[0]] :
      data_out[data[0]][data[1]] = []
      # data_idx[data[0]][data[1]] = []
    data_out[data[0]][data[1]].append(data)
    # data_idx[data[0]][data[1]].append(idx)

  with open(file_out, "w") as f :
    json.dump(data_out, f)
