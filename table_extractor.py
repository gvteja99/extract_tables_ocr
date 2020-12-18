import cv2
import pytesseract
from pytesseract import Output
from google.colab.patches import cv2_imshow

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from typing import Tuple, Union
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
import re
# img = cv2.imread("/content/fkey_table.png")

config_file = '/content/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = '/content/epoch_36.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Test a single image 
img_path = "/content/0155_081.png"

# img = "/content/TrigTable.png"

# Run Inference
result = inference_detector(model, img_path)

# Visualization results
# show_result_pyplot(img_path, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85)





# def rotate(
#         image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
# ) -> np.ndarray:
#     old_width, old_height = image.shape[:2]
#     angle_radian = math.radians(angle)
#     width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
#     height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     rot_mat[1, 2] += (width - old_width) / 2
#     rot_mat[0, 2] += (height - old_height) / 2
#     return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

# image = cv2.imread(img_path)
# grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# angle = determine_skew(grayscale)
# img = rotate(image, angle, (0, 0, 0))




img = cv2.imread(img_path)

d = pytesseract.image_to_data(img, output_type=Output.DICT)

n_boxes = len(d['text'])
boxes = []
for i in range(n_boxes):
    if int(d['conf'][i]) >= 0:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        boxes.append([x, y, x + w, y + h])
        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def overlap(rect1,rect2):
    try:
        p1 = Polygon([(rect1[0],rect1[1]), (rect1[0],rect1[3]),(rect1[2],rect1[3]),(rect1[2],rect1[1])])
        p2 = Polygon([(rect2[0],rect2[1]), (rect2[0],rect2[3]),(rect2[2],rect2[3]),(rect2[2],rect2[1])])
        return(p1.intersects(p2))
    except:
        return True


def bbx_limiter(bbx):
  for i in range(len(bbx)):
    if i%2 == 0:
      if bbx[i] < 0:
        bbx[i] = 0
      if bbx[i] > img.shape[1]:
        bbx[i] = img.shape[1]
    else:
      if bbx[i] < 0:
        bbx[i] = 0
      if bbx[i] > img.shape[0]:
        bbx[i] = img.shape[0]
  return bbx

def expand_bbox(bbox, increase = 0.1):
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  k = [None]*4
  k[0] = int(round(bbox[0] - (w*increase/4)))
  k[1] = int(round(bbox[1] - (h*increase/2)))
  k[2] = int(round(bbox[2] + (w*increase/4)))
  k[3] = int(round(bbox[3] + (h*increase/2)))

  return bbx_limiter(k)

table = result[0][2][0][:-1].round().astype("int32")

from shapely.geometry import Polygon
import numpy as np
ovlp_bxs = []
for i in boxes:

  
  if overlap(i, table) and i != [0, 0, img.shape[1], img.shape[0]]:
    # cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)
    ovlp_bxs.append(i)

# print("TABLE" ,table)

bb = np.append(np.array(ovlp_bxs).min(axis=0) [:2], np.array(ovlp_bxs).max(axis=0) [2:])

if bb[0] < table[0]:
  table[0] = bb[0]

if bb[1] < table[1]:
  table[1] = bb[1]

if bb[2] > table[2]:
  table[2] = bb[2]

if bb[3] > table[3]:
  table[3] = bb[3]

# exp_table = expand_bbox(table, increase = 0.1)
# cv2.rectangle(img, (table[0], table[1]), (table[2], table[3]), (0, 255, 0), 2)

# cv2_imshow(img)
exp_table = expand_bbox(table, increase = 0.05)


def inter_area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0



bbx_list = []
txt_list = []
for i in range(len(d["conf"])):
  if int(d["conf"][i]) > 0:
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    if w == img.shape[1] and h == img.shape[0]:
      continue
    if inter_area(exp_table, [x, y, x + w, y + h]) > 0:
      rem = re.sub(' +', ' ', d["text"][i]).strip()
      if rem != "" and rem != "|":
        # cv2_imshow(img[y:y+h, x:x+w])
        # print(d["text"][i], [x, y, x + w, y + h])
        # cv2.rectangle(img, (x ,y), (x+w, y+h), (0, 255, 0), 2)
        bbx_list.append([x, y, x + w, y + h])
        txt_list.append(d["text"][i])


arr = np.array(bbx_list)
order = np.argsort(arr[:, 0])
arr = arr[order]

tarr = np.array(txt_list)
tarr = tarr[order]

def in_list(item,L):
    for i in L:
        if item in i:
            return L.index(i)
    return -1


density = sum([(i[3] - i[1])*(i[2] - i[0]) for i in bbx_list])/((exp_table[3] - exp_table[1])*(exp_table[2] - exp_table[0]))

def find_closest(arr, tarr, gap = 0.4):
  closest_list = [None] * len(arr)
  for i in range(len(arr)):
    for j in range(i+1, len(arr)):
      y_i = arr[i][3] - arr[i][1]
      y_j = arr[j][3] - arr[j][1]
      x_limit = arr[i][2] + y_i + y_j
      uy_limit = arr[i][1] + y_i*gap
      ly_limit = arr[i][1] - y_i*gap
      if arr[i][2] < arr[j][0] and arr[j][0] < x_limit:
        if ly_limit < arr[j][1] and arr[j][1] < uy_limit:
        #   print(tarr[i], tarr[j])
          closest_list[i] = j
          
          continue
  return closest_list

if density < 0.2:
  res = find_closest(arr, tarr, gap = 0.4)
else:
  res = closest_list = [None] * len(arr)


red = arr.tolist()
red = [[i] for i in red]
ted = tarr.tolist()

def attach(i, bunch, res, ted):
  if res[i] == None:
    return bunch
  else:
    idx = res[i]
    res[i] = None
    bunch.append(idx)
    return attach(idx, bunch, res, ted)

# res = [3, None, 6, 2, None, 7, None , 4]

# ted = ["HI", "Hello", "world", "YOLO", "james", "bond", "red", "yellow"]

bunch = []
for i in range(len(res)):
  rid = attach(i, [], res, ted)
  if len(rid) != 0:
    rid.insert(0, i)
    bunch.append(rid)


def comb_bunch(red, ted, bunch):
  skip_idx = []
  for i in bunch:
    # print(i)
    idx = i[0]
    for j in i[1:]:
      skip_idx.append(j)
      red[idx].extend(red[j])
      text = " " + ted[j]
      ted[idx] += text
  brr = []
  trr = []
  for i in range(len(red)):
    red[i] = comb_bbx(red[i])
    if i not in skip_idx:
      brr.append(red[i])
      trr.append(ted[i])
  return brr, trr


def comb_bbx(med):
  xmin = None
  ymin = None
  xmax = None
  ymax = None
  for i in med:
    if xmin == None and ymin == None:
      xmin = i[0]
      ymin = i[1]
    else:
      if i[0] < xmin:
        xmin = i[0]
      if i[1] < ymin:
        ymin = i[1]

    if xmax == None and ymax == None:
      xmax = i[2]
      ymax = i[3]
    else:
      if i[2] > xmax:
        xmax = i[2]
      if i[3] > ymax:
        ymax = i[3]
  return [xmin, ymin, xmax, ymax]



red = arr.tolist()
red = [[i] for i in red]
ted = tarr.tolist()
brr, trr = comb_bunch(red, ted, bunch)

def circle(x1, y1, x2, y2, r1, r2): 
   
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  
    radSumSq = (r1 + r2) * (r1 + r2);  
    if (distSq >= radSumSq): 
        return False 
    else: 
        return True


def inputs(bbox):
  x = (bbox[0] + bbox[2])/2
  y = (bbox[1] + bbox[3])/2
  r = (bbox[2] - bbox[0])/2
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  mul = w/h

  if density < 0.1:
    if mul <= 3:
      r = r*mul
  return x, y, r

def find_closest_circle(brr, mul = 1.2):
  closest_list = []
  for i in range(len(brr)):
    for j in range(i+1, len(brr)):
      x1, y1, r1 = inputs(brr[i])
      x2, y2, r2 = inputs(brr[j])
      # print(brr[i], brr[j])
      # print(x1, y1, x2, y2, (r1*mul/2), (r2*mul/2))
      intersect = circle(x1, y1, x2, y2, r1*mul, r2*mul)
      # print(trr[i], trr[j], intersect, )
      # print(intersect)

      if intersect:
        # print(trr[i], trr[j])
        closest_list.append([i, j])
  return closest_list


if density < 0.2:
  result = find_closest_circle(brr, mul = 1.2)
else:
  result = find_closest_circle(brr, mul = 1)


import copy
test = copy.deepcopy(result)
skip_ids = []
for i in range(len(test)):
  if i in skip_ids:
    continue
  for j in range(i+1 , len(test)):
    if test[i][0] == test[j][0]:
      test[i].append(test[j][1])
      skip_ids.append(j)

rest = []
for i in range(len(test)):
  if i in skip_ids:
    continue
  else:
    rest.append(test[i])


fed = {}

for i in rest:
  fed[i[0]] = i[1:]

import networkx as nx
G = nx.Graph(fed)
nx.connected_components(G)

mist = [list(i) for i in nx.connected_components(G)]

# chains_1 = [sorted(set(i)) for i in mist]
chains = [sorted(set(i)) for i in mist]

def is_horizontal(i, j):
  m1 = (i[3] + i[1])/2
  m2 = (j[3] + j[1])/2

  if m1 <= m2:
    per = ((m2 - m1)/m1) * 100
  else:
    per = ((m1 - m2)/m2) * 100

  if per < 5:
    return 1
  else:
    return 0

# chains = []

# for k in chains_1:
#   md = []
#   for i in range(len(k)):
#     # if k[i] not in md:
#     #   md.append(k[i])
#     for j in range(i+1, len(k)):
#       if is_horizontal(brr[k[i]], brr[k[j]]) == 1:
#         print("hor")
#         md.append(j)
#   kd = []
#   for i in range(len(k)):
#     if i not in md:
#       kd.append(k[i])
#   chains.append(kd)


for i in chains:
  text_list = [trr[j] for j in i]
#   print(text_list)

cols = 0
for i in chains:
  if len(i) > cols:
    cols = len(i)

for i in chains:
  bbx_list = [brr[j] for j in i]
#   print(bbx_list)

def get_vlimits_bbx(bbox):
  ymin = img.shape[0]
  ymax = 0
  for i in bbox:
    # print(i)
    y1, y2 = i[1], i[3]
    if y1 < ymin:
      ymin = y1
    if y2 > ymax:
      ymax = y2
  return ymin, ymax


high_range = [0, 0]
for i in chains:
  if len(i) == cols:
    bbx_list = [brr[j] for j in i]
    # print("III", bbx_list)
    ymin, ymax = get_vlimits_bbx(bbx_list)
    # print(ymin, ymax)
    if ymax - ymin > high_range[1] - high_range[0]:
      high_range = [ymin, ymax]


yrange = high_range[1] - high_range[0]

kef = np.array(brr)
bff = np.array(brr)
tff = np.array(trr)

order = np.argsort(kef[:, 1])
min_y = kef[order][0][1]
order = np.argsort(kef[:, 3])
max_y = kef[order][-1][3]

spacing = int(round(yrange/cols))
above_cols = int(round((min_y - high_range[0])/spacing))
below_cols = int(round((max_y - high_range[1])//spacing))

cols = cols + above_cols + below_cols

col_spacing = []
for i in range(cols+1):
  y1 = max_y - spacing*(i)
  y2 = max_y - spacing*(i+1)
  range_col = [0, y2, img.shape[1]-1, y1]
  col_spacing.append(range_col)


def inter_area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

regions = []
for i in brr:
  max_area = 0

  ar_list = [inter_area(j, i) for j in col_spacing]
#   print(ar_list)
  idx = ar_list.index(max(ar_list))
  regions.append(idx)

from collections import Counter

rows = max(Counter(regions).values())


max_rows = []
count_reg = Counter(regions)
for i in sorted(count_reg.keys()):
  # print(i, count_reg[i])
  if count_reg[i] == rows:
    max_rows.append(i)


chain_rows = [None]* len(max_rows)
arr = np.array(regions)

for i in range(len(max_rows)):
  chain_rows[i] = np.where(arr==max_rows[i])

rows_list = []
for i in chain_rows:
  rows_list.append([brr[int(j)] for j in i[0]])
#   print([trr[int(j)] for j in i[0]])

ken = []
for i in rows_list:
  for j in i:
    ken.append(j)


ken = np.array(ken)
order = np.argsort(ken[:, 0])
xmin = ken[order][0][0]
xmin = img.shape[1]
xmax = 0 
for i in rows_list:
  for j in i:
    if j[0] < xmin:
      xmin = j[0]
    if j[2] > xmax:
      xmax = j[2]
      
min_idx1 = (np.where(ken[:,0]==xmin)[0][0])//rows
min_idx2 = (np.where(ken[:,0]==xmin)[0][0])%rows
min_row = rows_list[min_idx1][min_idx2]

max_idx1 = (np.where(ken[:,2]==xmax)[0][0])//rows
max_idx2 = (np.where(ken[:,2]==xmax)[0][0])%rows

max_row = rows_list[max_idx1][max_idx2]

ken = []
for i in brr:
  ken.append(i)

ken = np.array(ken)

bxmin = img.shape[1]
bxmax = 0 
for i in rows_list:
  for j in i:
    if j[0] < bxmin:
      bxmin = j[0]
    if j[2] > bxmax:
      bxmax = j[2]

bxmin = ken[np.argsort(ken[:, 0])][0][0]
bxmax = ken[np.argsort(ken[:, 2])][-1][2]


bmin_idx = (np.where(ken[:,0]==bxmin)[0][0])

bmin_row = brr[bmin_idx]

bmax_idx = (np.where(ken[:,2]==bxmax)[0][0])


bmax_row = brr[bmax_idx]

min_in_max = np.array(rows_list[max_idx1])
min_in_max = min_in_max[np.argsort(min_in_max[:, 0])]
max_mids = [(i[0]+i[2])/2 for i in min_in_max]

space_list = []
for i in range(len(max_mids)):
  if i == len(max_mids) - 1:
    break
  space_list.append(abs(max_mids[i+1] - max_mids[i]))

xspacing = int(round(sum(space_list)/len(space_list)))

def get_xdistance(a, b):
  x1 = (a[0] + a[2])/2
  y1 = (b[0] + b[2])/2

  return abs(y1 - x1)

left_rows =  int(round(get_xdistance(min_in_max[0], min_row)/xspacing))
max_in_min = np.array(rows_list[min_idx1])
max_in_min = max_in_min[np.argsort(max_in_min[:, 2])]

min_mids = [(i[0]+i[2])/2 for i in max_in_min]

space_list = []
for i in range(len(min_mids)):
  if i == len(min_mids) - 1:
    break
  space_list.append(abs(min_mids[i+1] - min_mids[i]))

xspacing_min = int(round(sum(space_list)/len(space_list)))
get_xdistance(max_in_min[-1], max_row)/xspacing_min
xspacing = int(round((xspacing +xspacing_min)/2))
right_rows =  int(round(get_xdistance(max_in_min[-1], max_row)/xspacing))
rows = rows + left_rows + right_rows
xspacing = int(round((xspacing +xspacing_min)/2))

if right_rows > 0:
  max_row = max_in_min[-1]

if left_rows > 0:
  min_row = min_in_max[0]

bleft_rows = int(round(get_xdistance(bmin_row, min_row)/xspacing))
bright_rows = int(round(get_xdistance(bmax_row, max_row)/xspacing))

if bright_rows > 0:
  max_row = bmax_row

if bleft_rows > 0:
  min_row = bmin_row

rows = rows + bleft_rows + bright_rows

bxmin = ken[np.argsort(ken[:, 0])][0][0]
bxmax = ken[np.argsort(ken[:, 2])][-1][2]
bymin = ken[np.argsort(ken[:, 1])][0][1]
bymax = ken[np.argsort(ken[:, 3])][-1][3]

max_x = bmax_row[2]
min_x = bmin_row[0]

max_y = bymax
min_y = bymin

temp = int(round((bmax_row[2] - bmin_row[0])/rows))
tempy = int(round((max_y - min_y)/cols))
rect_regions = []
rect_regions_idx = []
# order = np.argsort(kef[:, 3])
# max_y = kef[order][-1][3]
for i in range(cols):
  for j in range(rows):
    y2 = max_y - tempy*(i)
    x2 = max_x - temp*(j)
    y1 = y2 - tempy
    x1 = x2 - temp
    rect_regions.append([x1, y1, x2, y2])
    rect_regions_idx.append([i, j])

pt_brr = [[ int(round((i[0]+i[2])/2)) , int(round((i[1]+i[3])/2))] for i in brr]

def point_exists(pt, bbx):
  k = 0
  if pt[0] >= bbx[0] and pt[0] <= bbx[2]:
    if pt[1] >= bbx[1] and pt[1] <= bbx[3]:
      k = 1
  return k

def area_rect(a):
  return (a[3] - a[1]) * (a[2] - a[0])

def point_exists(pt, bbx):
  k = 0
  if pt[0] >= bbx[0] and pt[0] <= bbx[2]:
    if pt[1] >= bbx[1] and pt[1] <= bbx[3]:
      k = 1
  return k

pt_brr = [[ int(round((i[0]+i[2])/2)) , int(round((i[1]+i[3])/2))] for i in brr]

rect_index = [None] * len(rect_regions)
idx_list = [None] * len(rect_regions)
area_regions = []
point_regions = []
for k in range(len(rect_regions)):
  i = rect_regions[k]
  ar = [inter_area(i, j)/area_rect(j) for j in brr]
  area_regions.append(ar)

  mr = [point_exists(pt_brr[j], brr[j]) for j in range(len(brr))]
  point_regions.append(mr)

  idx = ar.index(max(ar))
  if max(ar) == 0:
    idx = None
  rect_index[k] = [idx, max(ar)]
  idx_list[k] = idx

def num_ovp(ar):
  ref = Counter(ar)
  return sum(ref.values()) - ref[0.0]



idx_list = [None] * len(rect_regions)

for i in range(len(area_regions)):
  ar = area_regions[i]
  if num_ovp(area_regions[i]) == 1:
    idx = ar.index(max(ar))
    idx_list[i] = idx
  else:
    idxs = [idx for idx, val in enumerate(area_regions[i]) if val != 0.0]
    for idx in idxs:
      if idx not in idx_list:
        if ar.index(max(ar)) == idx:
          idx_list[i] = idx


absent = []
for idx in range(len(brr)):
  if idx not in idx_list:
    absent.append(idx)

for i in absent:
  ar = []
  ir = []
  for j in range(len(area_regions)):
    if area_regions[j][i] != 0:
      ar.append(area_regions[j][i])
      ir.append(j)
    #   print(j, area_regions[j][i], idx_list[j])
      if idx_list[j] == None:
        idx_list[j] = i
        continue
  text = " " + trr[i]
  trr[idx_list[ir[ar.index(max(ar))]]] += text
    
def area_rect(a):
  return (a[3] - a[1]) * (a[2] - a[0])


ext = []
for i in Counter(idx_list).keys():
  if Counter(idx_list)[i] > 1 and i != None:
    ext.append(i)

ktm = np.array(rect_index) 
ftm = np.array(idx_list)

for i in ext:

  elf = np.where(ftm == i)[0]
  dlf = [rect_index[j][1] for j in elf]

  idx = dlf.index(max(dlf))
  for j in range(len(elf)):
    if j != idx:
    #   print("j")
    #   print(rect_index[elf[j]])
      rect_index[elf[j]] = [None, 0.0]
      idx_list[elf[j]] = None
    #   print(rect_index[elf[j]])

import pandas as pd
df = pd.DataFrame(index = [""]*cols, columns = [""]*rows )

for i in range(len(rect_regions_idx)):
  if idx_list[i] != None:

    # print(cols - 1 - rect_regions_idx[i][0], rows - 1 - rect_regions_idx[i][1], trr[idx_list[i]])
    r = cols - 1 - rect_regions_idx[i][0]
    c = rows - 1 - rect_regions_idx[i][1]
    txt = trr[idx_list[i]]

    df.iloc[r][c] = str(txt)
  else:

    r = cols - 1 - rect_regions_idx[i][0]
    c = rows - 1 - rect_regions_idx[i][1]
    txt = ""

    df.iloc[r][c] = str(txt)

df.to_csv("/content/0155_081.csv")