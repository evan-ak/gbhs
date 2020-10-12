
import json
import numpy
import random

import torch

import os
import datetime

from dataset import *
from mcgs import *
from model import *

datasets = {"train" : {"path" : "train1"},
            "valid1" : {"path" : "validation1"},
            "valid2" : {"path" : "validation2"},
            "test1" : {"path" : "test1", "save_to_csv" : True},
            "test2" : {"path" : "test2", "save_to_csv" : True}
            }

def train() :
  print("File: loading")
  local_path = os.path.abspath(".")
  for tag, data in datasets.items() :
    if "train" in tag or "valid" in tag :
      # with open(local_path + "/data/" + data["path"] + "/objects_detect_cp.json", "r") as f :
      with open(local_path + "/data/" + data["path"] + "/objects_cp.json", "r") as f :
        data["file_objects"] = json.load(f)
      with open(local_path + "/data/" + data["path"] + "/questions_grouped.json", "r") as f :
        data["file_questions"] = json.load(f)
      data["path_images"] = local_path + "/data/" + data["path"] + "/png_cp256/"
  print("File: loaded")

  print("Model: initializing")
  M_ActionNet = ActionNet().cuda()
  modules_meta_info = M_ActionNet.modules_meta_info
  print("Model: initialized\n")

  puzzle_types = []
  question_templates = {}
  for q_type in datasets["train"]["file_questions"].keys() :
    for f_type in datasets["train"]["file_questions"][q_type].keys() :
      puzzle_types.append((q_type, f_type))
      question_templates[(q_type, f_type)] = tuple([f_type] + datasets["train"]["file_questions"][q_type][f_type][0][3].split())

  save_results = {}
  datasets["valid"] = datasets["valid1"]
  print("Training on", datasets["train"]["path"])
  print("Validating on", datasets["valid"]["path"])
  for q_type, f_type in puzzle_types :
    print("================================")
    print("Question Type:", q_type, "Figure Type:", f_type)

    for tag, data in datasets.items() :
      if tag in ("train", ) :
        data["dataset"] = Dataset_FigureQA(data, q_type, f_type)
        data["datagetter"] = AutoDataGetter(data["dataset"], loader_param = {"batch_size": 256, "shuffle": True, "num_workers": 16})
      if tag in ("valid1", "valid2", ) :
        data["dataset"] = Dataset_FigureQA(data, q_type, f_type)
        data["datagetter"] = AutoDataGetter(data["dataset"], loader_param = {"batch_size": 512, "shuffle": True, "num_workers": 2})

    default_module =  M_ActionNet.reset_state(new_module_allowed = 3)
    all_MCGS_nodes = {}
    params_storaged = {}
    prior_actions = {}
    default_action = [default_module, -1, -1]
    prior_actions[tuple(default_action)] = {"actions_list" : [default_module, -1, -1], "prior_score" : 0.5}
    if len(save_results) > 0 :
      distances = {}
      for k in save_results.keys() :
        distances[k] = find_edit_distance(question_templates[k], question_templates[(q_type, f_type)])
      dis_sorted = sorted(distances.items(), key = lambda item:item[1])
      for k, _ in dis_sorted[:2] :
        nearby_action = save_results[k]["actions"]
        accus = [l[-1] for l in save_results[k]["finaleval"]]
        prior_actions[tuple(nearby_action)] = {"actions_list" : save_results[k]["actions"], "prior_score" : sum(accus) / len(accus)}
    for pa in prior_actions.values() :
      for m in pa["actions_list"] :
        if m >= 0 and M_ActionNet.modulelist[m].trainable is True :
          if not m in M_ActionNet.available_modules :
            M_ActionNet.available_modules.append(m)
      for m in pa["actions_list"] :
        if modules_meta_info[m]["level"] > 0 and not m in M_ActionNet.available_modules :
          M_ActionNet.available_modules.append(m)
      node = MCGSNode(actions_tree = None, actions_list = pa["actions_list"], prior_score = pa["prior_score"], all_nodes = all_MCGS_nodes, modules_meta_info = modules_meta_info)

    print("available module:", M_ActionNet.available_modules)

    N_MCGS = 1000
    weight_decay = (0.5, 0.25, 0.15, 0.1)
    bests = []

    for step in range(N_MCGS) :
      highest_score_in_d = [{}]
      total_score = {}
      for k, v in all_MCGS_nodes.items() :
        s = v.prior_score if v.score is None else v.score
        highest_score_in_d[-1][k] = s
        if v.visit_flag < 2 :
          total_score[k] = weight_decay[0] * s
      for i, w in enumerate(weight_decay[1:]) :
        highest_score_in_d.append({})
        for k in all_MCGS_nodes.keys() :
          scores = [highest_score_in_d[i][k]]
          for neighbor in all_MCGS_nodes[k].neighbors :
            key = str(neighbor.actions_list)
            if key in highest_score_in_d[i] :
              scores.append(highest_score_in_d[i][key])
            else :
              scores.append(all_MCGS_nodes[key].prior_score if all_MCGS_nodes[key].score is None else all_MCGS_nodes[key].score)
          max_score = max(scores)

          highest_score_in_d[-1][k] = max_score
          if k in total_score :
            total_score[k] += w * max_score
      for k in total_score.keys() :
        total_score[k] += 0.05 * 1.0 / (all_MCGS_nodes[k].visit_count + 1)

      with_random = random.random() < 0.5
      if with_random :
        for k in total_score.keys() :
          total_score[k] += 0.02 * random.random()

      node_to_explore = all_MCGS_nodes[max(total_score.items(), key = lambda item:item[1])[0]]
      print(q_type, f_type, "On Step", step, "exploring", node_to_explore.actions_list)

      if with_random :
        print("with random")
      score_sorted = sorted(total_score.items(), key = lambda item:item[1], reverse = True)
      for k, _ in score_sorted[:20] :
        print(format(total_score[k], ">8.4f"), end = "")
        for i in range(len(weight_decay)) :
          print(format(highest_score_in_d[i][k], ">8.4f"), end = "")
        print(format(all_MCGS_nodes[k].visit_count, ">4d"), k )

      exp_actions_list_str = str(node_to_explore.actions_list)
      if exp_actions_list_str in params_storaged :
        params = params_storaged[exp_actions_list_str]
      else :
        params = None
      eval_accuracy, eval_score = node_to_explore.explore(model = M_ActionNet, params = params, datasets = datasets)
      if eval_accuracy is not None :
        print("validation:", eval_accuracy, end = " ")
        add_to_stroage = False
        if exp_actions_list_str in params_storaged :
          for i in range(len(bests)) :
            if bests[i]["actions"] == node_to_explore.actions_list :
              if eval_score > bests[i]["score"] :
                add_to_stroage = True
                bests[i] = {"actions" : node_to_explore.actions_list, "accuracy" : eval_accuracy, "score" : eval_score}
              break
        elif len(bests) < 20 or eval_score > bests[-1]["score"] :
          add_to_stroage = True
          if len(bests) < 20 :
            bests.append({"actions" : node_to_explore.actions_list, "accuracy" : eval_accuracy, "score" : eval_score})
          else :
            params_storaged.pop(str(bests[-1]["actions"]))
            bests[-1] = {"actions" : node_to_explore.actions_list, "accuracy" : eval_accuracy, "score" : eval_score}
        if add_to_stroage :
          params = {}
          for a in node_to_explore.actions_list :
            if a >= 0 and M_ActionNet.modulelist[a].trainable and not a in params:
              params[a] = M_ActionNet.modulelist[a].state_dict()
              for k in params[a].keys() :
                params[a][k] = params[a][k].clone()
          params_storaged[exp_actions_list_str] = params
          bests = sorted(bests, key = lambda item:item["score"], reverse = True)

      if len(bests) > 0 :
        print("best:", bests[0]["actions"], bests[0]["accuracy"])
      else :
        print()
      print()

    best_action = bests[0]["actions"]
    best_action_tree = ActionsOps.list_to_tree(best_action, modules_meta_info)

    print("final training")
    M_ActionNet.train_modules(actions = best_action_tree, params = None, datasets = datasets, train_loops = range(3000),
                              lr = {0:1e-1, 1000:1e-2, 1500:1e-3, 2000:1e-4, 2500:1e-5},
                              optim = lambda p: torch.optim.SGD(p, lr=1e-1, momentum = 0.9, nesterov = True, weight_decay = 0.002),
                              N_train_log_step = 10)

    print("final validating")
    del datasets["train"]["datagetter"]
    del datasets["train"]["dataset"]
    del datasets["valid1"]["datagetter"]
    final_validate = []
    for k in ("valid", "valid2") :
      dataset = datasets[k]
      dataset["dataloader"] = torch.utils.data.DataLoader(dataset["dataset"], collate_fn = Dataset_FigureQA.collate_fn,
                                                          batch_size = 512, shuffle = False, num_workers = 4)
      predicts, answers, accuracy = M_ActionNet.test_action(best_action_tree, dataset = dataset)
      print(f"{k:>6} {int(accuracy*len(answers))} / {len(answers)}   {accuracy}")
      final_validate.append([k, int(accuracy*len(answers)), len(answers), accuracy])
      del dataset["dataloader"]
      del dataset["dataset"]

    for a in best_action :
      if a >= 0 and M_ActionNet.modulelist[a].trainable :
        M_ActionNet.modulelist[a].weight_frozen = True
    save_results[(q_type, f_type)] = {"actions" : best_action, "finaleval" : final_validate}

  logs = {"results":{str(k):v for k,v in save_results.items()}, "module":M_ActionNet.modulelist[-1].__str__()}
  str_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  path_save = local_path + "/saves/save_" + str_datetime
  with open(path_save + "_results.json", "w") as f :
    json.dump(logs, f)
  torch.save({i:m.state_dict() for i, m in enumerate(M_ActionNet.modulelist) if m.trainable and m.weight_frozen}, path_save + "_params.pt")

  return

def find_edit_distance(sen1, sen2) :
  len_x = len(sen1) + 1
  len_y = len(sen2) + 1
  matrix = numpy.zeros((len_x, len_y))
  for x in range(len_x):
    matrix[x, 0] = x
  for y in range(len_y):
    matrix[0, y] = y

  for x in range(1, len_x):
    for y in range(1, len_y):
      if sen1[x-1] == sen2[y-1]:
        matrix [x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1], matrix[x, y-1] + 1)
      else:
        matrix [x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1] + 1, matrix[x, y-1] + 1)
  return int(matrix[len_x - 1, len_y - 1])

if __name__ == "__main__" :
  print("\n================================")
  train()
  # test()
  print("================================\n")
