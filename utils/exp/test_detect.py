from layers.functions import Detect,PriorBox
from data.config import VOC_320
import torch

top_k=200
confidence_threshold = 0.5
nms_threshold = 0.45

priorbox = PriorBox(VOC_320)
detector = Detect(21, 0, top_k, confidence_threshold, nms_threshold)
with torch.no_grad():
    priors = priorbox.forward()
    loc = torch.randn(1,6375,4)
    conf = torch.randn(6375,21)
    arm_loc = torch.randn(1,6375,4)

out = detector.forward(loc, conf, priors, arm_loc_data=None)