"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path, PurePath
import os
import json

def json_save(json_res, json_path):
    ''' save all level sperated results as json file'''
    json_paths = []
    res = {
        'block_bbox': [],
        'line1_bbox': [],
        'line2_bezier': [],
    }
    for img_item in json_res:
        for k, v in img_item['results'].items():
            json_v = {
                    'shapes': [],
                    "imagePath": None,
                    "imageData": None,
                    "imageHeight": 512,
                    "imageWidth": 512
                }
            for i, (point, score) in enumerate(zip(v[0], v[1])):
                json_v['shapes'].append({
                    'points': point.cpu().numpy().tolist(),
                    'score': round(score.cpu().item(), 2),
                    'label': i,
                })
            json_v['imagePath'] = img_item['img_path'].split('/')[-1]
            res[k].append(json_v)
    for k, v in res.items():
        json_path_sub = os.path.join(json_path, k)
        os.makedirs(json_path_sub, exist_ok=True)
        for json_item in v:
            json_item_path = os.path.join(json_path_sub, json_item['imagePath'].split('.')[0] + '.json')
            with open(json_item_path, 'w') as f:
                json.dump(json_item, f)
        json_paths.append(json_path_sub)
    return json_paths

class Line12Block():
    def __init__(self):
        pass

    def calculate_intersection_area(self, bbox1, bbox2):
        """计算两个bbox的交集面积"""
        x1_min, y1_min = bbox1[0]
        x1_max, y1_max = bbox1[1]
        
        x2_min, y2_min = bbox2[0]
        x2_max, y2_max = bbox2[1]
        
        intersect_min_x = max(x1_min, x2_min)
        intersect_min_y = max(y1_min, y2_min)
        intersect_max_x = min(x1_max, x2_max)
        intersect_max_y = min(y1_max, y2_max)
        
        intersect_width = max(0, intersect_max_x - intersect_min_x)
        intersect_height = max(0, intersect_max_y - intersect_min_y)
        
        return intersect_width * intersect_height

    def calculate_area(self, bbox):
        """计算bbox的面积"""
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        return (x_max - x_min) * (y_max - y_min)

    def calculate_intersection_area_other(self, bbox1, bbox2):
        """计算两个bbox的交集面积"""
        x1_min, y1_min = bbox1[0], bbox1[1]
        x1_max, y1_max = bbox1[2], bbox1[3]
        
        x2_min, y2_min = bbox2[0], bbox2[1]
        x2_max, y2_max = bbox2[2], bbox2[3]
        
        intersect_min_x = max(x1_min, x2_min)
        intersect_min_y = max(y1_min, y2_min)
        intersect_max_x = min(x1_max, x2_max)
        intersect_max_y = min(y1_max, y2_max)
        
        intersect_width = max(0, intersect_max_x - intersect_min_x)
        intersect_height = max(0, intersect_max_y - intersect_min_y)
        
        return intersect_width * intersect_height

    def calculate_area_other(self, bbox):
        """计算bbox的面积"""
        x_min, y_min = bbox[0], bbox[1]
        x_max, y_max = bbox[2], bbox[3]
        return (x_max - x_min) * (y_max - y_min)

    def process(self, line_bbox_folder, region_bbox_folder, output_folder, filename):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历第一个文件夹
        # for filename in os.listdir(line_bbox_folder):
        if filename.endswith('.json'):
            line_bbox_path = os.path.join(line_bbox_folder, filename)
            region_bbox_path = os.path.join(region_bbox_folder, filename)
            if not os.path.exists(region_bbox_path):
                print(f"Failed Processed {filename}")
                # continue
                return
            # 读取行字符包围盒JSON文件
            with open(line_bbox_path, 'r') as file:
                line_data = json.load(file)
            
            # 读取区域字符包围盒JSON文件
            with open(region_bbox_path, 'r') as file:
                region_data = json.load(file)
            
            # 遍历每个行字符包围盒
            for line_shape in line_data['shapes']:
                line_bbox = line_shape['points']
                if len(line_bbox) == 2:
                    line_area = self.calculate_area(line_bbox)
                elif len(line_bbox) == 4:
                    line_area = self.calculate_area_other(line_bbox)
                # 遍历每个区域字符包围盒
                for region_shape in region_data['shapes']:
                    region_bbox = region_shape['points']
                    
                    # 计算交集面积
                    if len(line_bbox) == 2:
                        intersection_area = self.calculate_intersection_area(line_bbox, region_bbox)
                    elif len(line_bbox) == 4:
                        intersection_area = self.calculate_intersection_area_other(line_bbox, region_bbox)
                    
                    # 判断交集面积是否大于行字符包围盒面积的80%
                    if intersection_area >= 0.6 * line_area:
                        line_shape['region_label'] = region_shape['label']
                        break
                    else:
                        line_shape['region_label'] = None
            
            # 将结果写入新的JSON文件
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as file:
                json.dump(line_data, file, indent=4)
            print(f"Processed {filename}")

class Line22Line1():
    def __init__(self):
        pass
    def bezier_curve(self, p0, p1, p2, p3, t):
        """计算贝塞尔曲线在t点的值"""
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
        return x, y

    def calculate_center_line(self, upper_curve, lower_curve, num_samples=100):
        """计算贝塞尔曲线的中心线"""
        upper_points = [self.bezier_curve(*upper_curve, t) for t in np.linspace(0, 1, num_samples)]
        lower_points = [self.bezier_curve(*lower_curve, t) for t in np.linspace(0, 1, num_samples)]
        center_points = [( (up[0] + lp[0]) / 2, (up[1] + lp[1]) / 2) for up, lp in zip(upper_points, lower_points)]
        return center_points

    def is_point_in_bbox(self, point, bbox):
        """判断点是否在包围盒（bbox）中"""
        x, y = point
        (x_min, y_min), (x_max, y_max) = bbox
        return x_min <= x <= x_max and y_min <= y <= y_max

    def is_point_in_bbox_other(self, point, bbox):
        """判断点是否在包围盒（bbox）中"""
        x, y = point
        x_min, y_min, x_max, y_max = bbox
        return x_min <= x <= x_max and y_min <= y <= y_max

    def percentage_points_in_bbox(self, points, bbox):
        """计算中心线采样点在包围盒中的百分比"""
        if len(bbox) == 2:
            count = sum(1 for point in points if self.is_point_in_bbox(point, bbox))
        elif len(bbox) == 4:
            count = sum(1 for point in points if self.is_point_in_bbox_other(point, bbox))
        return count / len(points)

    def process(self, bezier_curve_folder, bbox_folder, output_folder, filename):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历第一个文件夹
        # for filename in os.listdir(bezier_curve_folder):
        if filename.endswith('.json'):
            bezier_curve_path = os.path.join(bezier_curve_folder, filename)
            bbox_path = os.path.join(bbox_folder, filename)
            if not os.path.exists(bbox_path):
                print(f"Failed Processed {filename}")
                # continue
                return
            # 读取贝塞尔曲线JSON文件
            with open(bezier_curve_path, 'r') as file:
                bezier_data = json.load(file)
            
            # 读取包围盒JSON文件
            with open(bbox_path, 'r') as file:
                bbox_data = json.load(file)
            
            # 遍历每个贝塞尔曲线
            for bezier_shape in bezier_data['shapes']:
                bezier_points = bezier_shape['points']
                
                # 上下两条贝塞尔曲线的控制点
                upper_curve = [(bezier_points[i], bezier_points[i + 1]) for i in range(0, 8, 2)]
                lower_curve = [(bezier_points[i], bezier_points[i + 1]) for i in range(8, 16, 2)]
                
                # 计算贝塞尔曲线的中心线
                center_line = self.calculate_center_line(upper_curve, lower_curve)
                
                # 遍历每个包围盒
                for bbox_shape in bbox_data['shapes']:
                    bbox_points = bbox_shape['points']
                    
                    # 计算采样点在包围盒中的百分比
                    percentage = self.percentage_points_in_bbox(center_line, bbox_points)
                    
                    # 判断中心线的采样点是否有60%以上在包围盒中
                    if percentage >= 0.6:
                        bezier_shape['bbox_label'] = bbox_shape['label']
                        break
                    else:
                        bezier_shape['bbox_label'] = None
            
            # 将结果写入新的JSON文件
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as file:
                json.dump(bezier_data, file, indent=4)
            print(f"Processed {filename}")
