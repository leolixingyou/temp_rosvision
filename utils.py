import os
import tensorrt as trt
import copy
import pycuda.driver as cuda
import pycuda.autoinit ## I had problem with this, so must import. Plz check yourself
import numpy as np
import cv2
import time
# import matplotlib.pyplot as plt

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def iou_yolo(box1, box2):
    ### if wanna run this file then use below two

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    union_area = area_box1 + area_box2 - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    return iou,box2

def jurdge_boxes(boxes,cls_id):
    for i, item1 in enumerate(boxes):
        for j, item2 in enumerate(boxes):
            if i != j:
                iou_source, box = iou_yolo(item1,item2)
                if iou_source > 0.5:
                    cls_id[i] = max(cls_id[i],cls_id[j]) 
    return cls_id

def filter_boxes(boxes,scores, cls_ids, fix_flag, conf=0.5):
    if fix_flag:
        fix_boxes=[]
        fix_cls=[]
        fix_scores=[]
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            if cls_id in [4,6,8,9,10,11,12,13,14]:
                score = scores[i]
                if score < conf:
                    continue
                fix_boxes.append(box)
                fix_cls.append(cls_id)
                fix_scores.append(score)
        return fix_boxes, fix_cls, fix_scores
    else:
        return boxes,cls_ids,scores

def get_box_info(img, boxes, scores, cls_ids, conf=0.2, class_names=None,da0="day"):

    box_result = []

    fix_list = [True,False]
    fix = fix_list[0]

    fix_boxes, fix_cls, fix_scores = filter_boxes(boxes, scores, cls_ids, fix_flag = fix,conf=conf)
    fix_cls = jurdge_boxes(fix_boxes,fix_cls)
    
    for i in range(len(fix_boxes)):

        box = fix_boxes[i]
        cls_id = int(fix_cls[i])
        score = fix_scores[i]

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        x0 = 0 if  x0 < 0 else x0
        y0 = 0 if  y0 < 0 else y0

        box_area = (x1-x0)*(y1-y0)
        ### filtering the boxes with area and position
        if box_area > 200 and x0 < 1700 and y1 > 59:
            box_result.append([cls_id,box_area,score,[x0,y0,x1,y1]])
            
    return  box_result

class BaseEngine(object):
    def __init__(self, engine_path):
        self.rgb_day_list = [(148, 108, 138), (207, 134, 240), (138, 88, 55), (26, 99, 86), (38, 125, 80), (72, 80, 57), # 6 
                        (118, 23, 200), (117, 189, 183), (0, 128, 128), (60, 185, 90), (20, 20, 150), (13, 131, 204), 
                        (30, 200, 200), (43, 38, 105), (104, 235, 178), (135, 68, 28), (140, 202, 15), (67, 115, 220),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30),(30, 80, 30)]

        self.mean = None
        self.std = None
        self.n_classes = 80

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    ### image steam
    def steam_inference(self, origin_img, conf=0.5, end2end=False,day_night='day'):
        print('DETECTION...')
        t1 =time.time()
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        print('Pre-process is :',round((time.time()-t1)*1000,2),' ms')
        t3 = time.time()
        data = self.infer(img)
        print('Model is :',round((time.time()-t3)*1000,2),' ms')
        t2 = time.time()
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            box_result = get_box_info(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names,da0=day_night)
        print('Post-process is :',round((time.time()-t2)*1000,2),' ms')
        
        return box_result
    

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def draw_img(self, img, boxes, fcolor, class_names):
        height, weight, _ = img.shape
        tl = 3 or round(0.002 * (height + weight) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        
        cur_img = copy.copy(img)

        if len(boxes) > 0 :
            for i in range(len(boxes)):

                box = boxes[i][3]
                cls_id = boxes[i][0]
                score = boxes[i][2]

                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                x0 = 0 if  x0 < 0 else x0
                y0 = 0 if  y0 < 0 else y0

                _COLORS = self.rgb_day_list
                
                c1, c2 = (x0,y0), (x1,y1)
                cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], thickness=tl, lineType=cv2.LINE_AA)
                tf = max(tl - 1, 1)  # font thickness
                text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
                t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(cur_img, c1, c2, _COLORS[cls_id], -1, cv2.LINE_AA)  # filled
                cv2.putText(cur_img, text, (c1[0], c1[1] - 2), 0, tl / 3, fcolor, thickness=tf, lineType=cv2.LINE_AA)

        img = cur_img
        return img