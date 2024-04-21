import os
import time
import math
import numpy as np
import copy
import cv2
import argparse
import pycuda.driver as cuda ## I had problem with this, so must import. Plz check yourself 
import pycuda.autoinit ## I had problem with this, so must import. Plz check yourself

from detection.det_infer import Predictor

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose

###
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
###
from sort import *

class Camemra_Node:
    def __init__(self,args,day_night):
        rospy.init_node('Camemra_node')
        self.args = args
        
        # self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'green3_h', 'bus',
        #                       'red3_h', 'truck', 'yellow3_h', 'green4_h', 'red4_h', 'yellow4_h',
        #                       'redgreen4_h', 'redyellow4_h', 'greenarrow4_h', 'red_v', 'yellow_v', 'green_v','black']
        ## Changable
        self.class_names = ['person', 'bicycle','car','bus','motorcycle','truck', 'green', 'red', 'yellow',
                            'red_arrow', 'red_yellow', 'green_arrow','green_yellow','green_right',
                            'warn','black','tl_v', 'tl_p', 'traffic_sign', 'warning', 
                            'tl_bus']
        self.n_classes = len(self.class_names)

        ### The point which can be refered for find out the mid traffic light
        self.baseline_boxes = [960,270]

        sort_max_age = 15
        sort_min_hits = 2
        sort_iou_thresh = 0.1
        self.sort_tracker_f60 = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
        
        self.get_f60_new_image = False
        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_f60_img = {'img':None, 'header':None}
        self.bbox_f60 = PoseArray()
        
        ### Det Model initiolization
        self.day_night = day_night
        self.det_pred = Predictor(engine_path=args.det_weight , day_night=day_night)
        
        self.pub_od_f60 = rospy.Publisher('/mobinha/perception/camera/bounding_box', PoseArray, queue_size=1)
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback)

        ########### pub result image ###############
        self.pub_f60_det = rospy.Publisher('/det_result/f60', Image, queue_size=1)
      
        self.bridge = CvBridge()
        self.is_save =False
        self.sup = []
        ########### pub result image ###############
       
        ##########################
        ########filter part#######

        self.real_cls_hist = 0

        ########filter part#######
        ##########################


    ##########################
    ########filter part#######

    def filtered_obs(self,traffic_light_obs):
        new_obs = []
        for obs in traffic_light_obs:
            if obs[0] == 18 and self.real_cls_hist != 0:
                obs[0] = self.real_cls_hist
            new_obs.append(obs)
        return new_obs

    ########filter part#######
    ##########################

    def get_one_boxes(self,traffic_light_obs):
    ##########################
    ########filter part#######

        traffic_light_obs = self.filtered_obs(traffic_light_obs)

    ########filter part#######
    ##########################
        if len(traffic_light_obs) >0:
            # print(f'traffic_light_obs is {traffic_light_obs}')            
            boxes = np.array(traffic_light_obs)[:,3]
            ## calculate the distance from the baseline point TO GET MID target
            distances = [math.sqrt(((box[0] + box[2]) / 2 - self.baseline_boxes[0]) ** 2 + ((box[1] + box[3]) / 2 - self.baseline_boxes[1]) ** 2) for box in boxes]
            areas = [((box[2] - box[0]) * (box[3] - box[1])) for box in boxes]
            weights = [0.6*(distances[x] / max(distances)) + 0.4*(1 - (areas[x] / max(areas)))  for x in range(len(boxes))]
            result_box = [traffic_light_obs[weights.index(min(weights))]]
            return result_box
        else:
            result_box = traffic_light_obs
            return result_box

    def get_traffic_light_objects(self,bbox_f60):
        traffic_light_obs = []

        if len(bbox_f60) > 0:
            for traffic_light in bbox_f60:
                if traffic_light[2] > 0.2:  # if probability exceed 20%
                    traffic_light_obs.append(traffic_light)
        # sorting by size
        traffic_light_obs = self.get_one_boxes(traffic_light_obs)
        return traffic_light_obs

    def IMG_f60_callback(self,msg):
        if not self.get_f60_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.cur_f60_img['img'] = front_img
            self.cur_f60_img['header'] = msg.header
            self.get_f60_new_image = True

    def pose_set(self,bboxes,flag):
        bbox_pose = PoseArray()
        for bbox in bboxes:
            pose = Pose()
            pose.position.x = bbox[0]# box class
            pose.position.y = bbox[1]# box area
            pose.position.z = bbox[2]# box score
            pose.orientation.x = bbox[3][0]# box mid x
            pose.orientation.y = bbox[3][1]# box mid y
            pose.orientation.z = bbox[3][2]# box mid y
            pose.orientation.w = bbox[3][3]# box mid y
            bbox_pose.poses.append(pose)

        if flag == 'f60':
            self.pub_od_f60.publish(bbox_pose)
       
    def det_pubulissher(self,det_img,det_box,flag):
        if flag =='f60':
            det_f60_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_f60_det.publish(det_f60_msg)
    
    def update_tracking(self,box_result,flag):
        update_list = []
        if len(box_result)>0:
            cls_id = np.array(box_result)[:,0]
            areas = np.array(box_result)[:,1]
            scores = np.array(box_result)[:,2]
            boxes = np.array(box_result)[:,3]
            dets_to_sort = np.empty((0,6))

            for i,box in enumerate(boxes):
                x0, y0, x1, y1 = box
                cls_name = cls_id[i]
                dets_to_sort = np.vstack((dets_to_sort, 
                            np.array([x0, y0, x1, y1, scores[i], cls_name])))
            if flag == 'f60':
                tracked_dets = self.sort_tracker_f60.update(dets_to_sort)
                tracks = self.sort_tracker_f60.getTrackers()

            bbox_xyxy = tracked_dets[:,:4]
            categories = tracked_dets[:, 4]

            new_areas = (bbox_xyxy[:,2] - bbox_xyxy[:,0]) * (bbox_xyxy[:,3] - bbox_xyxy[:,1])
            update_list = [[int(categories[x]),new_areas[x],scores[x],bbox_xyxy[x]] for x in range(len(tracked_dets)) ]

        else:
            tracked_dets = self.sort_tracker_f60.update()

        return update_list

    def image_process(self,img,flag):
        if flag == 'f60' :
            ### using with vs
            box_result_f60 = self.det_pred.steam_inference(img,conf=0.1, end2end='end2end' ,day_night=self.day_night)
            ### using shell file named 'vision.sh'
            # box_result_f60 = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end,day_night=self.day_night)

            box_result_f60 = self.update_tracking(box_result_f60,flag)
            det_img_60 = self.det_pred.draw_img(img,box_result_f60, [255, 255, 255])
            tl_boxes = self.get_traffic_light_objects(box_result_f60)
            filter_img_f60 = self.det_pred.draw_img(det_img_60, tl_boxes, [0, 0, 0], self.class_names)
            
            self.det_pubulissher(filter_img_f60, tl_boxes,flag)
            

    def main(self):
        rate = rospy.Rate(150)
        while not rospy.is_shutdown():
            if self.get_f60_new_image:
                self.sub_f60_img['img'] = self.cur_f60_img['img']
                orig_im_f60 = copy.copy(self.sub_f60_img['img']) 
                self.image_process(orig_im_f60,'f60')
                self.get_f60_new_image = False

            rate.sleep()
          

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--end2end", default=False, action="store_true",help="use end2end engine")
    
    day_night_list = ['day','night']
    day_night = day_night_list[0]
    if day_night == 'day':
        # parser.add_argument('--det_weight', defaulssst="./detection/weights/epiton_3_nms.trt")  ### end2end 
        # parser.add_argument('--det_weight', default="./detection/weights/230615_songdo_day_no_nms.trt")  ### end2end 
        # parser.add_argument('--det_weight', default="./detection/weights/yolov7x_flicker_with_nms.trt")  ### end2end  
        # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/yolov7x_flicker.trt")  ### end2end
        # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/incheon_dataset.trt")  ### end2end 
        # parser.add_argument('--det_weight', default="./detection/weights/weight_trt/new.trt")  ### end2end  dayeong_on_ioniq
        parser.add_argument('--det_weight', default="/workspace/ros_vision/detection/weights/onnx/integrate_each_class_fp16_0314.trt")  ### no end2end xingyou  

    if day_night == 'night':
        print('*'*12)
        print('*** NIGHT TIME ***')
        print('*'*12)
        parser.add_argument('--det_weight', default="./detection/weights/230615_night_songdo_no_nms_2.trt")  ### end2end

    args = parser.parse_args()
    
    camemra_node = Camemra_Node(args,day_night)
    camemra_node.main()

