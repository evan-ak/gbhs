
import os
import math
import json

local_path = os.path.abspath(".")
_sets = ["train1", "validation1", "validation2"]
x_std = 256
y_std = 256
with open(local_path + "/color_dict.json", "r") as f :
  color_dict = json.load(f)

mode = "cp"
assert mode in ["rm", "ex", "cp"]

for _set in _sets :
  file_in = local_path + "/" + _set + "/annotations.json"
  file_out = local_path + "/" + _set + "/objects_" + mode + ".json"

  with open(file_in, "r") as f :
    data_in = json.load(f)
  data_out = []
  for image in data_in :
    x0 = image["general_figure_info"]["figure_info"]["bbox"]["bbox"]["w"]
    y0 = image["general_figure_info"]["figure_info"]["bbox"]["bbox"]["h"]

    image_data = []
    if image["type"] in ['vbar_categorical', 'hbar_categorical'] :
      if len(image["models"]) > 1 :
        print("warn")
      model = image["models"][0]
      N = len(model["labels"])
      for i in range(N) :
        object_data = []
        object_data.append(math.floor(model["bboxes"][i]["x"] + model["bboxes"][i]["w"]/2))
        object_data.append(math.floor(model["bboxes"][i]["y"] + model["bboxes"][i]["h"]/2))
        object_data.append(math.floor(model["bboxes"][i]["x"]))
        object_data.append(math.floor(model["bboxes"][i]["y"]))
        object_data.append(math.ceil(model["bboxes"][i]["w"]))
        object_data.append(math.ceil(model["bboxes"][i]["h"]))
        object_data.append(1)
        object_data.append(model["colors"][i])
        if not model["labels"][i] in color_dict :
          color_dict[model["labels"][i]] = model["colors"][i]
        image_data.append(object_data)
    elif image["type"] in ['line', 'dot_line'] :
      for model in image["models"] :
        for b in model["bboxes"] :
          object_data = []
          object_data.append(math.floor(b["x"] + b["w"]/2))
          object_data.append(math.floor(b["y"] + b["h"]/2))
          object_data.append(math.floor(b["x"]))
          object_data.append(math.floor(b["y"]))
          object_data.append(math.floor(b["w"]))
          object_data.append(math.floor(b["h"]))
          object_data.append(1)
          object_data.append(model["color"])
          image_data.append(object_data)
    elif image["type"] in ['pie'] :
      for model in image["models"] :
        object_data = []
        object_data.append(math.floor(model["bbox"]["x"] + model["bbox"]["w"]/2))
        object_data.append(math.floor(model["bbox"]["y"] + model["bbox"]["h"]/2))
        object_data.append(math.floor(model["bbox"]["x"]))
        object_data.append(math.floor(model["bbox"]["y"]))
        object_data.append(math.floor(model["bbox"]["w"]))
        object_data.append(math.floor(model["bbox"]["h"]))
        object_data.append(1)
        if model["label"] in color_dict :
          object_data.append(color_dict[model["label"]])
        else :
          print("undefined color !!!")
        image_data.append(object_data)

    if True :
      figure = image["general_figure_info"]
      title = figure["title"]
      object_data = []
      object_data.append(math.floor(title["bbox"]["x"] + title["bbox"]["w"]/2))
      object_data.append(math.floor(title["bbox"]["y"] + title["bbox"]["h"]/2))
      object_data.append(math.floor(title["bbox"]["x"]))
      object_data.append(math.floor(title["bbox"]["y"]))
      object_data.append(math.ceil(title["bbox"]["w"]))
      object_data.append(math.ceil(title["bbox"]["h"]))
      object_data.append(0)
      object_data.append(title["text"])
      image_data.append(object_data)

    if image["type"] in ['vbar_categorical', 'hbar_categorical', 'line', 'dot_line'] :
      xaxis = figure["x_axis"]["label"]
      object_data = []
      object_data.append(math.floor(xaxis["bbox"]["x"] + xaxis["bbox"]["w"]/2))
      object_data.append(math.floor(xaxis["bbox"]["y"] + xaxis["bbox"]["h"]/2))
      object_data.append(math.floor(xaxis["bbox"]["x"]))
      object_data.append(math.floor(xaxis["bbox"]["y"]))
      object_data.append(math.ceil(xaxis["bbox"]["w"]))
      object_data.append(math.ceil(xaxis["bbox"]["h"]))
      object_data.append(0)
      object_data.append(xaxis["text"])
      image_data.append(object_data)

      xlabel = figure["x_axis"]["major_labels"]
      for i in range(len(xlabel["values"])//2) :
        object_data = []
        object_data.append(math.floor(xlabel["bboxes"][i]["x"] + xlabel["bboxes"][i]["w"]/2))
        object_data.append(math.floor(xlabel["bboxes"][i]["y"] + xlabel["bboxes"][i]["h"]/2))
        object_data.append(math.floor(xlabel["bboxes"][i]["x"]))
        object_data.append(math.floor(xlabel["bboxes"][i]["y"]))
        object_data.append(math.ceil(xlabel["bboxes"][i]["w"]))
        object_data.append(math.ceil(xlabel["bboxes"][i]["h"]))
        object_data.append(0)
        object_data.append(xlabel["values"][i])
        image_data.append(object_data)

      yaxis = figure["y_axis"]["label"]
      object_data = []
      object_data.append(math.floor(yaxis["bbox"]["x"] + yaxis["bbox"]["w"]/2))
      object_data.append(math.floor(yaxis["bbox"]["y"] + yaxis["bbox"]["h"]/2))
      object_data.append(math.floor(yaxis["bbox"]["x"]))
      object_data.append(math.floor(yaxis["bbox"]["y"]))
      object_data.append(math.ceil(yaxis["bbox"]["w"]))
      object_data.append(math.ceil(yaxis["bbox"]["h"]))
      object_data.append(0)
      object_data.append(yaxis["text"])
      image_data.append(object_data)

      ylabel = figure["y_axis"]["major_labels"]
      for i in range(len(ylabel["values"])//2) :
        object_data = []
        object_data.append(math.floor(ylabel["bboxes"][i]["x"] + ylabel["bboxes"][i]["w"]/2))
        object_data.append(math.floor(ylabel["bboxes"][i]["y"] + ylabel["bboxes"][i]["h"]/2))
        object_data.append(math.floor(ylabel["bboxes"][i]["x"]))
        object_data.append(math.floor(ylabel["bboxes"][i]["y"]))
        object_data.append(math.ceil(ylabel["bboxes"][i]["w"]))
        object_data.append(math.ceil(ylabel["bboxes"][i]["h"]))
        object_data.append(0)
        object_data.append(ylabel["values"][i])
        image_data.append(object_data)

    if image["type"] in ['line', 'pie', 'dot_line'] :
      for item in image["general_figure_info"]["legend"]["items"] :
        object_data = []
        object_data.append(math.floor(item["preview"]["bbox"]["x"] + item["preview"]["bbox"]["w"]/2))
        object_data.append(math.floor(item["preview"]["bbox"]["y"] + item["preview"]["bbox"]["h"]/2))
        object_data.append(math.floor(item["preview"]["bbox"]["x"]))
        object_data.append(math.floor(item["preview"]["bbox"]["y"]))
        object_data.append(math.ceil(item["preview"]["bbox"]["w"]))
        object_data.append(math.ceil(item["preview"]["bbox"]["h"]))
        object_data.append(1)
        object_data.append(color_dict[item["model"]])
        image_data.append(object_data)

        object_data = []
        object_data.append(math.floor(item["label"]["bbox"]["x"] + item["label"]["bbox"]["w"]/2))
        object_data.append(math.floor(item["label"]["bbox"]["y"] + item["label"]["bbox"]["h"]/2))
        object_data.append(math.floor(item["label"]["bbox"]["x"]))
        object_data.append(math.floor(item["label"]["bbox"]["y"]))
        object_data.append(math.ceil(item["label"]["bbox"]["w"]))
        object_data.append(math.ceil(item["label"]["bbox"]["h"]))
        object_data.append(0)
        object_data.append(item["label"]["text"])
        image_data.append(object_data)

    if mode == "ex" :
      x_ex = (x_std - x0) // 2
      y_ex = (y_std - y0) // 2
      for object_data in image_data :
        object_data[0] += x_ex
        object_data[1] += y_ex
        object_data[2] += x_ex
        object_data[3] += y_ex
    elif mode == "cp" :
      x_cp = x_std / x0
      y_cp = y_std / y0
      for object_data in image_data :
        object_data[0] = math.floor(object_data[0] * x_cp)
        object_data[1] = math.floor(object_data[1] * y_cp)
        object_data[2] = math.floor(object_data[2] * x_cp)
        object_data[3] = math.floor(object_data[3] * y_cp)
        object_data[4] = math.ceil(object_data[4] * x_cp)
        object_data[5] = math.ceil(object_data[5] * y_cp)

    data_out.append(image_data)

  with open(file_out, "w") as f :
    json.dump(data_out, f)
