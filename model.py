
import math
import torch

from mcgs import ActionsOps
from densenet import DenseNet, DenseNetSP

datatype = ["none", "object", "objects", "text", "image", "answer"]
imagesize = 256

class PolicyNet(torch.nn.Module) :
  pass

class ActionNet(torch.nn.Module) :
  def __init__(self) :
    super(ActionNet, self).__init__()
    self.modulelist = []
    # 0 1
    self.modulelist.extend([ModuleLookFor(match_idx = 0), ModuleLookFor(match_idx = 1)])
    # 2 3 4 5
    self.modulelist.extend([ModuleLookUp(), ModuleLookDown(), ModuleLookLeft(), ModuleLookRight()])
    # 6
    self.modulelist.extend([ModuleFindSameColor()])
    # 7 +
    for _ in range(40) :
      self.modulelist.append(ModuleJudgerDense())

    self.modules_meta_info = {}
    for i, m in enumerate(self.modulelist) :
      self.modules_meta_info[i] = { "level" : 0,
                                    "shape" : m.module_shape,
                                    "data_type_in" : m.data_type_in,
                                    "data_type_out" : m.data_type_out}
    self.modules_meta_info[-2] = {"level" : 0, "shape" : (1,0), "data_type_in" : ("answer",), "data_type_out": None}
    self.modules_meta_info[-1] = {"level" : 0, "shape" : (0,1), "data_type_in" : None, "data_type_out": "none"}

    self.modulelist = torch.nn.ModuleList(self.modulelist)
    self.available_modules = []
    self.batch_data = None

  def forward_step(self, action_node) :
    if action_node.action == -1 :
      return None
    else :
      module_inputs = []
      for c in action_node.children :
        module_inputs.append(self.forward_step(c))
      module_inputs.append(self.batch_data)
      return self.modulelist[action_node.action](*module_inputs)

  def forward(self, actions) :
    if actions.action > -2 :
      return self.forward_step(actions)
    else :
      return self.forward_step(actions.children[0])

  def train_modules(self, actions = None, params = None, datasets = None, train_loops = range(0), lr = {0:2e-3}, optim = torch.optim.Adam, N_train_log_step = -1, N_eval_log_step = -1) :
    assert actions is not None
    assert len(train_loops) > 0

    module_in_optimizer = []
    params_list = []
    actions_list = ActionsOps.tree_to_list(actions, [])
    t = [0,0]
    logs = [[],[]]
    for a in actions_list :
      if a >= 0 and self.modulelist[a].trainable and not self.modulelist[a].weight_frozen and not a in module_in_optimizer:
        module_in_optimizer.append(a)
        params_list.append({"params": self.modulelist[a].parameters()})
        if params is not None and a in params :
          self.modulelist[a].load_state_dict(params[a])
        else :
          self.modulelist[a].reset_params()
    for k, v in lr.items() :
      if train_loops[0] >= k :
        base_lr = v
      else :
        break
    if len(module_in_optimizer) > 0 :
      optimizer = optim(params_list)
      print("Training on", module_in_optimizer, "in", actions_list)
    else :
      print("Not to train")
      return self.eval_action(actions, datasets["valid"])

    F_loss = torch.nn.CrossEntropyLoss(reduction = "none")
    lr_adjust = lr
    final_eval_accuracy = None
    for loop in train_loops :
      if loop in lr_adjust.keys() :
        for g in optimizer.param_groups :
          g["lr"] = lr_adjust[loop]

      keys, objs, img_T, ans_T = datasets["train"]["datagetter"].get()
      self.batch_data = { "keys" : keys,
                          "objects" : objs,
                          "images" : img_T,
                          "failed" : [],
                          "ftype" : datasets["train"]["dataset"].f_type }

      y = self(actions)
      failed_filter = torch.ones(len(ans_T)).cuda()
      failed_filter[self.batch_data["failed"]] = 0.0
      optimizer.zero_grad()
      loss = (F_loss(y, ans_T) * failed_filter.detach()).mean()
      loss.backward()
      optimizer.step()
      need_train_log = N_train_log_step > 0 and (loop + 1) % N_train_log_step == 0
      need_eval_log = N_eval_log_step > 0 and (loop + 1) % N_eval_log_step == 0

      if need_train_log or need_eval_log :
        print("\t\t", format(loop + 1, ">3d"), end = "")
        if need_train_log :
          _, result = y.max(1)
          result[self.batch_data["failed"]] = -1
          accuracy = (result == ans_T).sum().item() / len(ans_T)
          logs[0].append(loss.item())
          logs[1].append(accuracy)
          print("  Loss:", format(loss.item(), ">16.10f"), "  Accuracy:", format(accuracy, ">6.4f"), end="")
        if need_eval_log :
          accuracy = self.eval_action(actions, datasets["valid"])
          print("  validation:", format(accuracy, ">6.4f"), end = "")
          final_eval_accuracy = accuracy
        print()
    return final_eval_accuracy

  def eval_action(self, actions, dataset) :
    self.eval()
    keys, objs, img_T, ans_T = dataset["datagetter"].get()
    self.batch_data = { "keys" : keys,
                        "objects" : objs,
                        "images" : img_T,
                        "failed" : [],
                        "ftype" : dataset["dataset"].f_type }
    with torch.no_grad() :
      y = self(actions)
    self.train()
    _, result = y.max(1)
    result[self.batch_data["failed"]] = -1
    accuracy = (result == ans_T).sum().item() / len(ans_T)

    return accuracy

  def test_action(self, actions, dataset) :
    self.eval()
    results = []
    answers = []
    for keys, objs, img_T, ans_T in dataset["dataloader"] :
      img_T = img_T.cuda()
      ans_T = ans_T.cuda()
      self.batch_data = { "keys" : keys,
                          "objects" : objs,
                          "images" : img_T,
                          "failed" : [],
                          "ftype" : dataset["dataset"].f_type }
      with torch.no_grad() :
        y = self(actions)
      _, result = y.max(1)
      result[self.batch_data["failed"]] = -1
      results.append(result)
      answers.append(ans_T)
    self.train()
    results = torch.cat(results)
    answers = torch.cat(answers)
    if (answers >= 0).sum().item() > 0 :
      accuracy = ((answers >= 0) * (results == answers)).sum().item() / (answers >= 0).sum().item()
    else :
      accuracy = None

    return results, answers, accuracy

  def reset_state(self, new_module_allowed = -1) :
    for m in self.modulelist :
      if m.trainable :
        m.weight_frozen_tmp = False
    self.available_modules = []
    default_module = None
    for i, m in enumerate(self.modulelist) :
      if m.trainable is False :
        self.available_modules.append(i)
      elif m.weight_frozen is False and default_module is None :
        default_module = i
    self.available_modules.append(default_module)
    return default_module

class ModuleLookFor(torch.nn.Module) :
  def __init__(self, match_idx = 0) :
    super(ModuleLookFor, self).__init__()
    self.module_name = "look_for"
    self.module_shape = (1,1)
    self.data_type_in = (("text", "none"), "objects", "text")
    self.data_type_out = "object"
    self.match_idx = match_idx
    self.trainable = False

  def forward(self, t, w) :
    if t is None :
      t = list(w["keys"])
    assert len(t) == len(w["objects"])
    y = []
    for i in range(len(t)) :
      y.append(None)
      if i in w["failed"] :
        continue
      if type(t[i]) in (tuple, list) :
        if len(t[i]) > 1 :
          t[i] = t[i][self.match_idx]
        else :
          t[i] = t[i][0]
      for obj in w["objects"][i] :
        if obj[-1] == t[i] :
          y[-1] = obj
          break
      if y[-1] is None :
        w["failed"].append(i)
    return y

class ModuleLookUp(torch.nn.Module) :
  def __init__(self) :
    super(ModuleLookUp, self).__init__()
    self.module_name = "look_up"
    self.module_shape = (1,1)
    self.data_type_in = (("object",), "objects")
    self.data_type_out = "object"
    self.trainable = False

  def forward(self, x, w) :
    assert len(x) == len(w["objects"])
    y = []
    for i in range(len(x)) :
      y.append(None)
      if i in w["failed"] :
        continue
      dmin = 0
      for obj in w["objects"][i] :
        dx = math.fabs(obj[0] - x[i][0])
        dy = obj[1] - x[i][3]
        if dy < 0 and - dy > 1.0 * dx :
          if y[-1] is None or dx < dmin :
            y[-1] = obj
            dmin = dx
      if y[-1] is None :
        y[-1] = x[i]
    return y

class ModuleLookDown(torch.nn.Module) :
  def __init__(self) :
    super(ModuleLookDown, self).__init__()
    self.module_name = "look_down"
    self.module_shape = (1,1)
    self.data_type_in = (("object",), "objects")
    self.data_type_out = "object"
    self.trainable = False

  def forward(self, x, w) :
    assert len(x) == len(w["objects"])
    y = []
    for i in range(len(x)) :
      y.append(None)
      if i in w["failed"] :
        continue
      dmin = 0
      for obj in w["objects"][i] :
        dx = math.fabs(obj[0] - x[i][0])
        dy = obj[1] - x[i][3] - x[i][5]
        if dy > 0 and dy > 1.0 * dx :
          if y[-1] is None or dx < dmin :
            y[-1] = obj
            dmin = dx
      if y[-1] is None :
        y[-1] = x[i]
    return y

class ModuleLookLeft(torch.nn.Module) :
  def __init__(self) :
    super(ModuleLookLeft, self).__init__()
    self.module_name = "look_left"
    self.module_shape = (1,1)
    self.data_type_in = (("object",), "objects")
    self.data_type_out = "object"
    self.trainable = False

  def forward(self, x, w) :
    assert len(x) == len(w["objects"])
    y = []
    for i in range(len(x)) :
      y.append(None)
      if i in w["failed"] :
        continue
      dmin = 0
      for obj in w["objects"][i] :
        dx = obj[0] - x[i][2]
        dy = math.fabs(obj[1] - x[i][1])
        if dx < 0 and - dx > 1.0 * dy :
          dy += math.fabs(dx)/5
          if y[-1] is None or dy < dmin :
            y[-1] = obj
            dmin = dy
      if y[-1] is None :
        y[-1] = x[i]
    return y

class ModuleLookRight(torch.nn.Module) :
  def __init__(self) :
    super(ModuleLookRight, self).__init__()
    self.module_name = "look_right"
    self.module_shape = (1,1)
    self.data_type_in = (("object",), "objects")
    self.data_type_out = "object"
    self.trainable = False

  def forward(self, x, w) :
    assert len(x) == len(w["objects"])
    y = []
    for i in range(len(x)) :
      y.append(None)
      if i in w["failed"] :
        continue
      dmin = 0
      for obj in w["objects"][i] :
        dx = obj[0] - x[i][2] - x[i][4]
        dy = math.fabs(obj[1] - x[i][1])
        if dx > 0 and dx > 1.0 * dy :
          if y[-1] is None or dy < dmin :
            y[-1] = obj
            dmin = dy
      if y[-1] is None :
        y[-1] = x[i]
    return y

class ModuleFindSameColor(torch.nn.Module) :
  def __init__(self) :
    super(ModuleFindSameColor, self).__init__()
    self.module_name = "find_same_color"
    self.module_shape = (1,1)
    self.data_type_in = (("object", "objects"), "objects")
    self.data_type_out = "objects"
    self.trainable = False

  def forward(self, x, w) :
    assert len(x) == len(w["objects"])
    y = []
    for i in range(len(x)) :
      if len(x[i]) <= 0 :
        w["failed"].append(i)
      if i in w["failed"] :
        y.append(None)
        continue
      if type(x[i][0]) in (tuple, list) :
        x[i] = x[i][0]
      y.append([])
      for obj in w["objects"][i] :
        if x[i][-1] == obj[-1] and not (w["ftype"] in ("2", "4") and x[i][0] == obj[0] and x[i][1] == obj[1]) :
          y[-1].append(obj)
    return y

class ModuleJudgerDense(torch.nn.Module) :
  def __init__(self) :
    super(ModuleJudgerDense, self).__init__()
    self.module_name = "judger"
    self.module_shape = (2, 1)
    self.data_type_in = (("object", "objects", "none"), ("object", "objects", "none"), "image")
    self.data_type_out = "answer"
    self.trainable = True
    self.weight_frozen = False
    self.weight_frozen_tmp = False
    self.color_refs = {}
    self.independent_branch = True
    if self.independent_branch :
      self.densenet = DenseNetSP()
    else :
      self.densenet = DenseNet()

  def forward(self, x, y, w) :
    imgs = w["images"]
    imgs_masked = [[],[]]
    objs = [x, y]
    mask_ones = torch.ones(3, imagesize, imagesize).cuda()
    for i in range(len(objs)) :
      if objs[i] is not None :
        assert len(objs[i]) == len(imgs)
        for j in range(len(objs[i])) :
          mask = torch.zeros(3, imagesize, imagesize).cuda()
          if j in w["failed"] :
            pass
          elif objs[i][j] is None :
            w["failed"].append(j)
            pass
          elif type(objs[i][j]) in (tuple, list) and len(objs[i][j]) > 0 :
            if not type(objs[i][j][0]) in (tuple, list) :
              objs[i][j] = [objs[i][j]]
            for obj in objs[i][j] :
              obj = obj.copy()
              while obj[4] <= 0 :
                obj[2] -= 1
                obj[4] += 2
              while obj[5] <= 0 :
                obj[3] -= 1
                obj[5] += 2

              if w["ftype"] in ("3", ) :
                if obj[-1][0] == "#" :
                  if obj[-1] in self.color_refs :
                     color_ref = self.color_refs[obj[-1]]
                  else :
                    color_ref = [int(obj[-1][5-k*2:7-k*2], 16)/255.0 for k in range(3)]
                    color_ref = torch.FloatTensor(color_ref).cuda().view(-1,1,1).expand(3, imagesize, imagesize)
                    self.color_refs[obj[-1]] = color_ref
                  color_match = (imgs[j][:, obj[3]:obj[3]+obj[5], obj[2]:obj[2]+obj[4]] == color_ref[:, obj[3]:obj[3]+obj[5], obj[2]:obj[2]+obj[4]]).sum(0) == 3
                  mask[:, obj[3]:obj[3]+obj[5], obj[2]:obj[2]+obj[4]] = color_match.unsqueeze(0).expand(3,-1,-1)
              else :
                mask[:, obj[3]:obj[3]+obj[5], obj[2]:obj[2]+obj[4]] = mask_ones[:, obj[3]:obj[3]+obj[5], obj[2]:obj[2]+obj[4]]
          else :
            pass
          imgs_masked[i].append(mask * (mask_ones - imgs[j]))
      else :
        imgs_masked[i] = [(mask_ones - img) for img in imgs]
      imgs_masked[i] = torch.stack(imgs_masked[i], 0)

    if self.independent_branch :
      y = self.densenet(*imgs_masked)
    else :
      y = self.densenet(torch.cat(imgs_masked, 1))
    return y

  def reset_params(self) :
    self.densenet.reset_params()
