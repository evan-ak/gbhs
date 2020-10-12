
import cv2
import torch

class Dataset_FigureQA(torch.utils.data.Dataset) :
  def __init__(self, data = None, q_type = "", f_type = "") :
    self.q_type = q_type
    self.f_type = f_type
    self.questions = data["file_questions"][q_type][f_type]
    self.objects = data["file_objects"]
    self.path_images = data["path_images"]
    self.N_questions = len(self.questions)
    self.need_CP = not "cp" in self.path_images

  def __getitem__(self, index) :
    q = self.questions[index]
    keys = (q[5], q[6])
    img_idx = q[2]
    objs = self.objects[img_idx]
    if self.need_CP :
      img = cv2.resize(cv2.imread(self.path_images + str(img_idx) + ".png"), (imagesize, imagesize)).transpose(2,0,1) / 255.0
    else :
      img = cv2.imread(self.path_images + str(img_idx) + ".png").transpose(2,0,1) / 255.0
    img_T = torch.FloatTensor(img)
    ans = q[4] if q[4] is not None else -1
    ans_T = torch.LongTensor([ans])
    return keys, objs, img_T, ans_T

  def __len__(self) :
    return self.N_questions

  @staticmethod
  def collate_fn(data) :
    zipped = list(zip(*data))
    zipped[2] = torch.stack(zipped[2])
    zipped[3] = torch.cat(zipped[3])
    return zipped

class AutoDataGetter() :
  def __init__(self, dataset = None, loader_param = {}, sava_data = False, repeat = True) :
    self.dataset = dataset
    self.loader_param = loader_param
    self.dataloader = torch.utils.data.DataLoader(self.dataset, collate_fn = Dataset_FigureQA.collate_fn, **self.loader_param)
    self.dataiter = iter(self.dataloader)
    self.sava_data = sava_data
    self.data_saved = None
    self.repeat = repeat

  def get(self) :
    if self.sava_data and self.data_saved is not None :
      return self.data_saved
    else :
      try:
        data = self.dataiter.next()
      except StopIteration:
        if not self.repeat :
          return None
        self.dataiter = iter(self.dataloader)
        data = self.dataiter.next()
      data = [d.cuda() if type(d) is torch.Tensor else d for d in data]
      if self.sava_data :
        self.data_saved = data
      return data

  def __del__(self) :
    del self.dataloader
