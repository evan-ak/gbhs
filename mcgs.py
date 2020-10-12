
import numpy
import torch

actions_length_limit = 9
ADD_TYPE_ILLEGAL_ACTIONS = False
ADD_STRUCT_ILLEGAL_ACTIONS = False

class MCGSNode() :
  def __init__(self, actions_tree = None, actions_list = None, actions_legality = True, neighbors = None, prior_score = 0, all_nodes = None, modules_meta_info = None) :
    self.modules_meta_info = modules_meta_info
    self.actions_tree = actions_tree
    self.actions_list = actions_list
    if self.actions_tree is not None and self.actions_list is None :
      self.actions_list = ActionsOps.tree_to_list(self.actions_tree, [])
    elif self.actions_list is not None :
      self.actions_tree = ActionsOps.list_to_tree(self.actions_list, modules_meta_info)
    self.actions_legality = actions_legality
    self.visit_flag = 0
    self.visit_count = 0
    self.train_count = 0
    self.train_once = 50
    self.train_total = 250
    self.neighbors = []
    self.neighbors_visit_count = 0
    self.accuracy = None
    self.score = None
    if prior_score < 0 :
      self.prior_score = 0.5 + ActionsOps.find_dev_score_bias(self.actions_list, modules_meta_info)
    else :
      self.prior_score = prior_score
    self.prior_score += ActionsOps.find_len_score_bias(self.actions_list)

    self.all_MCGS_nodes = all_nodes
    self.all_MCGS_nodes[str(self.actions_list)] = self
    self.prior_explore_estimation = numpy.random.rand(len(self.actions_list))
    self.explore_flag = [1] * len(self.actions_list)

  def explore(self, model, params, datasets) :
    assert sum(self.explore_flag) > 0
    modules_meta_info = self.modules_meta_info

    explore_estimation = numpy.copy(self.prior_explore_estimation) * self.explore_flag
    idx_e = explore_estimation.argmax()
    self.explore_flag[idx_e] = 0
    self.visit_count += 1
    if self.visit_count == len(self.actions_list) :
      self.visit_flag = 2
    else :
      self.visit_flag = 1

    if self.train_count < self.train_total :
      train_target = self.train_count + self.train_once if self.visit_flag < 2 else self.train_total
      eval_accuracy = model.train_modules(actions = self.actions_tree, params = params, datasets = datasets,
                                          train_loops = range(self.train_count, train_target),
                                          lr = {0:1e-1},
                                          optim = lambda p: torch.optim.SGD(p, lr=1e-1, momentum = 0.9, nesterov = True, weight_decay = 0.002),
                                          N_train_log_step = 5, N_eval_log_step = 10)
      self.train_count = train_target
      self.accuracy = eval_accuracy
      self.score = self.accuracy + 0.01 * (actions_length_limit - len(self.actions_list))
    else :
      eval_accuracy = None

    print("exploring at", self.actions_list, ":", idx_e)

    _, node_e = ActionsOps.list_to_tree(self.actions_list, modules_meta_info, mark_node = idx_e)
    node_e_child_idx = -1
    for i, c in enumerate(node_e.parent.children) :
      if c is node_e :
        node_e_child_idx = i
        break
    assert node_e_child_idx >= 0
    new_attempts = []

    # ADD
    if len(self.actions_list) < actions_length_limit :
      for i in model.available_modules :
        if not modules_meta_info[i]["data_type_out"] \
                in modules_meta_info[node_e.parent.action]["data_type_in"][node_e_child_idx] :
          if not ADD_TYPE_ILLEGAL_ACTIONS :
            continue
        for j in range(modules_meta_info[i]["shape"][0]) :
          attempt_legality = True
          if not modules_meta_info[i]["data_type_out"] \
                  in modules_meta_info[node_e.parent.action]["data_type_in"][node_e_child_idx] :
            attempt_legality = False
          else :
            for k in range(modules_meta_info[i]["shape"][0]) :
              if not modules_meta_info[node_e.action]["data_type_out"] in modules_meta_info[i]["data_type_in"][k] :
                attempt_legality = False
                break
          if not attempt_legality :
            if not ADD_TYPE_ILLEGAL_ACTIONS :
              continue

          root_new_attempt, node_add_before = ActionsOps.list_to_tree(self.actions_list, modules_meta_info, mark_node = idx_e)
          node_add_after = node_add_before.parent
          node_add = ActionNode(i, node_add_after)
          for k, c in enumerate(node_add_after.children) :
            if c is node_add_before :
              node_add_after.children[k] = node_add
              break
          for k in range(modules_meta_info[i]["shape"][0]) :
            node_add.children.append(node_add_before if j == k else ActionNode(-1, node_add))

          new_attempts.append((ActionsOps.tree_to_list(root_new_attempt, []), attempt_legality))

    # DELETE
    if node_e.action >= 0 :
      for i in range(modules_meta_info[node_e.action]["shape"][0]) :
        attempt_legality = True
        if not modules_meta_info[node_e.children[i].action]["data_type_out"] \
                in modules_meta_info[node_e.parent.action]["data_type_in"][node_e_child_idx] :
          attempt_legality = False
        if not attempt_legality :
          if not ADD_TYPE_ILLEGAL_ACTIONS :
            continue

        root_new_attempt, node_delete = ActionsOps.list_to_tree(self.actions_list, modules_meta_info, mark_node = idx_e)
        for k, c in enumerate(node_delete.parent.children) :
          if c is node_delete :
            node_delete.parent.children[k] = node_delete.children[i]
            node_delete.children[i].parent = node_delete.parent
            break

        new_attempts.append((ActionsOps.tree_to_list(root_new_attempt, []), attempt_legality))

    # CHANGE
    if node_e.action >= 0 :
      for i in model.available_modules :
        if modules_meta_info[i]["level"] > 0 and len(self.actions_list) + len(advanced_actions[i]) - 1 > actions_length_limit :
          continue
        if i == node_e.action :
          continue
        attempt_legality = True
        if modules_meta_info[node_e.action]["shape"] != modules_meta_info[i]["shape"] :
          attempt_legality = False
          if not ADD_STRUCT_ILLEGAL_ACTIONS :
            continue
        elif not modules_meta_info[i]["data_type_out"] in \
                  modules_meta_info[node_e.parent.action]["data_type_in"][node_e_child_idx] :
          attempt_legality = False
        else :
          for j in range(modules_meta_info[node_e.action]["shape"][0]) :
            if not modules_meta_info[node_e.children[j].action]["data_type_out"] in \
                    modules_meta_info[i]["data_type_in"][j] :
              attempt_legality = False
              break
        if not attempt_legality :
          if not ADD_TYPE_ILLEGAL_ACTIONS :
            continue

        root_new_attempt, node_change = ActionsOps.list_to_tree(self.actions_list, modules_meta_info, mark_node = idx_e)
        node_change.action = i

        new_attempts.append((ActionsOps.tree_to_list(root_new_attempt, []), attempt_legality))

    i = 0
    if len(new_attempts) > 0 :
      existing_MCGS_nodes = []
      new_MCGS_nodes = []
      while True :
        flag = False
        for j in range(i) :
          if new_attempts[j][0] == new_attempts[i][0] :
            new_attempts.pop(i)
            i -= 1
            flag = True
            break
        if not flag :
          if str(new_attempts[i][0]) in self.all_MCGS_nodes :
            existing_MCGS_nodes.append(new_attempts[i])
          else :
            new_MCGS_nodes.append(new_attempts[i])
        i += 1
        if i >= len(new_attempts) :
          break

      for n_n in new_MCGS_nodes :
        new_node = MCGSNode(actions_tree = None, actions_list = n_n[0], prior_score = -1, all_nodes = self.all_MCGS_nodes, modules_meta_info = modules_meta_info)
        if ADD_TYPE_ILLEGAL_ACTIONS :
          new_node.actions_legality = n_n[1]
        new_node.neighbors.append(self)
        self.neighbors.append(new_node)
        print(n_n[0], n_n[1])
      for n_n in existing_MCGS_nodes :
        existing_node = self.all_MCGS_nodes[str(n_n[0])]
        if not self in existing_node.neighbors :
          existing_node.neighbors.append(self)
        if not existing_node in self.neighbors :
          self.neighbors.append(existing_node)

    return eval_accuracy, self.score

class ActionNode() :
  def __init__(self, action = None, parent = None) :
    self.action = action
    self.parent = parent
    self.children = []

  def __str__(self) :
    return "action: " + str(self.action) + " parent: " + ("None" if self.action == -2 else str(self.parent.action)) + " children: " + str([c.action for c in self.children])

class ActionsOps() :
  @staticmethod
  def list_to_tree(actions_list, modules_meta_info, mark_node = None) :
    root = ActionNode(-2, None)
    visit = root
    nodes = []
    for i,a in enumerate(actions_list) :
      nodes.append(ActionNode(a, visit))
      visit.children.append(nodes[-1])
      if a >= 0 :
        visit = visit.children[-1]
      while len(visit.children) == modules_meta_info[visit.action]["shape"][0] :
        visit = visit.parent
        if visit is root :
          if mark_node is None:
            return root
          else :
            return root, nodes[mark_node]
    if mark_node is None:
      return root
    else :
      return root, nodes[mark_node]

  @staticmethod
  def tree_to_list(node_visit, acts_list) :
    if node_visit.action > -2 :
      acts_list.append(node_visit.action)
    for c in node_visit.children :
      ActionsOps.tree_to_list(c, acts_list)
    return acts_list

  @staticmethod
  def advanced_compress(actions_list, mark_index = None) :
    actions_list = actions_list.copy()
    patterns = [[v, k] for k, v in advanced_actions.items()]
    if len(patterns) > 0 :
      patterns = sorted(patterns, key = lambda item:len(item[0]), reverse = True)
    else :
      if mark_index is None :
        return actions_list
      else :
        return actions_list, mark_index
    i = 0
    while i < len(patterns) :
      i = 0
      while i < len(patterns) :
        p = patterns[i][0]
        l = len(p)
        match = False
        for s in range(0, len(actions_list) - l + 1) :
          if actions_list[s:s+l] == p :
            match = True
            if mark_index is not None :
              if mark_index >= s+l :
                mark_index -= (l - 1)
              elif mark_index >=s :
                mark_index = s
            actions_list = actions_list[:s] + [patterns[i][1]] + actions_list[s+l:]
            break
        if match :
          break
        else :
          i += 1
    if mark_index is None :
      return actions_list
    else :
      return actions_list, mark_index

  @staticmethod
  def advanced_decompress(actions_list) :
    actions_list = actions_list.copy()
    i = 0
    while i < len(actions_list) :
      a = actions_list[i]
      if a in advanced_actions :
        actions_list = actions_list[:i] + advanced_actions[a].copy() + actions_list[i+1:]
        i += len(advanced_actions[a])
      else :
        i += 1
    return actions_list

  @staticmethod
  def find_std_dev(actions_list, modules_meta_info) :
    branches = [1]
    for a in actions_list :
      branches[-1] += 1
      if a == -1 or modules_meta_info[a]["shape"][0] > 1 :
        branches.append(0)
    branches = branches[:-1]
    std = numpy.std(branches)
    return std / (len(actions_list) + 1)

  @staticmethod
  def find_len_score_bias(actions_list, alpha = 0.5) :
    return alpha / len(actions_list)

  @staticmethod
  def find_dev_score_bias(actions_list, modules_meta_info, alpha = -0.3) :
    return alpha * ActionsOps.find_std_dev(actions_list, modules_meta_info)
