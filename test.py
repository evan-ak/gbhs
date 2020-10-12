
import json
import numpy
import random

import torch

import os
import pandas

from dataset import *
from mcgs import *
from model import *


datasets = {"test1" : {"path" : "test1", "save_to_csv" : True},
            "test2" : {"path" : "test2", "save_to_csv" : True}
            }
path_load = "/home/evan/Shiratsuyu/publish/saves/fin"

def test() :
  print("File: loading")
  local_path = os.path.abspath(".")
  for tag, data in datasets.items() :
    if "test" in tag :
      with open(local_path + "/data/" + data["path"] + "/objects_detect_cp.json", "r") as f :
      # with open(local_path + "/data/" + data["path"] + "/objects_cp.json", "r") as f :
        data["file_objects"] = json.load(f)
      with open(local_path + "/data/" + data["path"] + "/questions_grouped.json", "r") as f :
        data["file_questions"] = json.load(f)
      data["path_images"] = local_path + "/data/" + data["path"] + "/png_cp256/"
      if not "save_to_csv" in data :
        data["save_to_csv"] = False
      if data["save_to_csv"] :
        data["path_csv"] = local_path + "/data/" + data["path"] + "/out.csv"
        data["path_questions_raw"] = local_path + "/data/" + data["path"] + "/qa_pairs.json"
        data["path_questions_idx"] = local_path + "/data/" + data["path"] + "/qa_pairs.json"
  with open(path_load + "_results.json", "r") as f :
    save_results = json.load(f)
  save_results = {eval(k):v for k,v in save_results["results"].items()}
  save_params = torch.load(path_load + "_params.pt")
  print("File: loaded")

  print("Model: initializing")
  M_ActionNet = ActionNet().cuda()
  modules_meta_info = M_ActionNet.modules_meta_info
  for k, v in save_params.items() :
    M_ActionNet.modulelist[k].load_state_dict(v)
    M_ActionNet.weight_frozen = True
  print("Model: initialized\n")

  for tag, data in datasets.items() :
    corrects = 0
    total = 0
    if data["save_to_csv"] :
      with open(data["path_questions_raw"], "r") as f :
        questions_raw = json.load(f)["qa_pairs"]
      N = len(questions_raw)
      answers_csv = { "question_index" : [i for i in range(N)] ,
                      "image_index" : [questions_raw[i]["image_index"] for i in range(N)],
                      "question_id" : [questions_raw[i]["question_id"] for i in range(N)] ,
                      "question_string" : [questions_raw[i]["question_string"] for i in range(N)] ,
                      "answer": [1 for i in range(N)] }

    print("Testng on", data["path"])
    for q_type in data["file_questions"].keys() :
      for f_type in data["file_questions"][q_type].keys() :
        print("================================")
        print("Question Type:", q_type, "Figure Type:", f_type)

        actions_list = save_results[(q_type, f_type)]["actions"]
        data["dataset"] = Dataset_FigureQA(data, q_type, f_type)
        data["dataloader"] = torch.utils.data.DataLoader(data["dataset"], collate_fn = Dataset_FigureQA.collate_fn,
                                                         batch_size = 512, shuffle = False, num_workers = 4)
        predicts, answers, accuracy = M_ActionNet.test_action(ActionsOps.list_to_tree(actions_list, modules_meta_info), dataset = data)

        print(len(answers), end = " ")
        if accuracy is not None :
          print(accuracy)
          corrects += len(answers) * accuracy
          total += len(answers)
        else :
          print()

        if data["save_to_csv"] :
          predicts = predicts.cpu().tolist()
          for i in range(len(answers)) :
            p = int(predicts[i]) if predicts[i] >= 0 else 1
            answers_csv["answer"][data["file_questions"][q_type][f_type][i][-1]] = p

    print("================================")
    if total > 0 :
      print(total, corrects / total)
    if data["save_to_csv"] :
      dataframe = pandas.DataFrame(answers_csv)
      dataframe.to_csv(data["path_csv"], index = False, sep = ",")
    print()

if __name__ == "__main__" :
  print("\n================================")
  test()
  print("================================\n")
