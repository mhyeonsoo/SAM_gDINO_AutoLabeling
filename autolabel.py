# python3 autolabel.py /path/to/target_dir \
# --config_file GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py \
# --checkpoint GroundingDINO/weights/groundingdino_swinb_cogcoor.pth \
# --device cuda \
# --text_prompt "cauliflower . broccoli . zucchini" \
# --box_threshold 0.30 \
# --text_threshold 0.25 \
# --iou_threshold 0.8

#!/usr/bin/env python3
import argparse
import os
import copy
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO 관련 모듈
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# 그 외 모듈
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as TT

# COCO Annotation 관련 (segmentation, bbox 계산)
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import datetime
# (pycocotools는 COCO 평가 시 사용 – 여기서는 json 저장만 하므로 직접 사용하지 않음)

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


########################################
# 1. 이미지 로드 및 전처리 함수 (블럭 2)
########################################
def load_image(image_path):
    """
    이미지 파일을 로드하여 PIL 이미지와 Grounding DINO 전처리에 맞는 텐서를 반환합니다.
    """
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)  # 3 x H x W 텐서
    return image_pil, image_tensor


########################################
# 2. 모델 로드 함수 (블럭 3)
########################################
def load_model(model_config_path, model_checkpoint_path, device):
    """
    모델 config 파일과 체크포인트를 이용해 모델을 로드합니다.
    """
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("모델 로드 결과:", load_res)
    model.eval()
    return model


########################################
# 3. 추론 함수 (블럭 4)
########################################
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """
    이미지와 텍스트 프롬프트(caption)를 입력받아 모델 추론을 수행한 후,
    임계값을 적용하여 박스와 예측된 phrase 리스트를 반환합니다.
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # 임계값으로 filtering
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]  # num_filt x 256
    boxes_filt = boxes[filt_mask]    # num_filt x 4

    # 모델 내 tokenizer를 사용하여 phrase 추출
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


########################################
# 4. 시각화를 위한 함수 (블럭 5, 6)
########################################
def show_mask(mask, ax, random_color=True):
    """
    디버깅/시각화를 위해 마스크를 표시합니다.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    """
    디버깅/시각화를 위해 박스를 그리고 label을 표시합니다.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label, fontsize=12, color='green')


########################################
# 5. COCO Annotation 관련 함수 (블럭 13)
########################################
def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    """
    단일 객체의 서브 마스크(2D binary mask)를 입력받아 COCO segmentation annotation을 생성합니다.
    """
    # sub_mask 에서 contour 추출 (객체 경계)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    segmentations = []
    polygons = []
    for contour in contours:
        # (row, col) -> (x, y) 변환 및 좌표 보정
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
    # 다수의 polygon 결합하여 bbox 및 area 계산
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = [x, y, width, height]
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }
    return annotation


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """
    단일 이미지에 대한 COCO image 정보(dict)를 생성합니다.
    image_size는 (width, height) 형식이어야 합니다.
    """
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }
    return image_info


def IOUcalc(registered, cand_area, thresh):
    """
    이미 등록된 bbox들과 IoU를 계산하여, 임계값보다 크면 False (중복) 반환.
    """
    for bbox in registered:
        iou = get_iou(bbox, cand_area)
        if iou >= float(thresh):
            return False
    return True


def get_iou(bb1, bb2):
    """
    두 bbox (dict, keys: 'x1','x2','y1','y2')의 IoU를 계산합니다.
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


########################################
# 6. Main 함수: 디렉토리 내 이미지 순회 및 처리
########################################
def main():
    parser = argparse.ArgumentParser(
        description="디렉토리 내 이미지들을 대상으로 Grounding DINO를 이용해 객체 검출 및 COCO annotation JSON 생성"
    )
    parser.add_argument("target_dir", type=str, help="이미지 파일들이 위치한 디렉토리")
    parser.add_argument("--config_file", type=str,
                        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                        help="모델 config 파일 경로")
    parser.add_argument("--checkpoint", type=str,
                        default="GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
                        help="모델 checkpoint 파일 경로")
    parser.add_argument("--device", type=str, default="cuda",
                        help="사용할 device (cuda 또는 cpu)")
    parser.add_argument("--text_prompt", type=str,
                        default="cauliflower . broccoli . zucchini",
                        help="텍스트 프롬프트")
    parser.add_argument("--box_threshold", type=float, default=0.30,
                        help="박스 임계값")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="텍스트 임계값")
    parser.add_argument("--iou_threshold", type=float, default=0.8,
                        help="IOU 임계값 (중복 bbox 제거용)")
    args = parser.parse_args()

    # 모델 로드 (한 번만 로드)
    print("모델 로드 중...")
    model = load_model(args.config_file, args.checkpoint, args.device)
    print("모델 로드 완료.")

    # COCO annotation에 사용할 카테고리 사전 (블럭 14)
    CAT_ID = {'asparagus': 1, 'broccoli': 2, 'carrot': 3, 'cauliflower': 4, 'potato': 5, 'zucchini': 6}

    # 대상 디렉토리 내의 이미지 파일 순회
    for file in os.listdir(args.target_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(args.target_dir, file)
            print("\n처리 중:", image_path)
            try:
                image_pil, image_tensor = load_image(image_path)
            except Exception as e:
                print(f"이미지 로드 실패: {e}")
                continue
            width, height = image_pil.size

            # Grounding DINO 모델 추론
            boxes_filt, pred_phrases = get_grounding_output(
                model, image_tensor, args.text_prompt,
                args.box_threshold, args.text_threshold,
                with_logits=True, device=args.device
            )
            if boxes_filt.shape[0] == 0:
                print("검출된 박스 없음.")
                continue

            # (블럭 11) 각 박스 영역에 대해 binary mask 생성
            masks = []
            for box in boxes_filt:
                # box: [x_min, y_min, x_max, y_max] (정수형 변환)
                box_list = list(map(int, box.tolist()))
                x_min, y_min, x_max, y_max = box_list
                mask = np.zeros((height, width), dtype=np.uint8)
                # 이미지 범위 내로 보정
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width, x_max)
                y_max = min(height, y_max)
                mask[y_min:y_max, x_min:x_max] = 1
                masks.append(mask)

            if len(masks) == 0:
                print("생성된 mask가 없습니다.")
                continue

            ########################################
            # COCO annotation 구성 (블럭 14)
            ########################################
            coco_annotation = {"images": [], "annotations": [], "categories": []}
            # 배경 카테고리 추가
            coco_annotation["categories"].append({"supercategory": None, "id": 0, "name": "_background_"})
            # CAT_ID에 정의된 카테고리 추가
            for i, category in enumerate(CAT_ID.keys()):
                coco_annotation["categories"].append({"supercategory": None, "id": i+1, "name": category})
            print("사용 카테고리:", coco_annotation["categories"])

            image_id = 0
            annotation_id = 0
            is_crowd = 0
            registered_regions = []  # 중복 bbox 제거용

            # 각 검출된 객체에 대해 annotation 생성
            for i, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
                # box는 tensor → dict 변환 (x1, y1, x2, y2)
                bbox = {'x1': box[0].item(), 'y1': box[1].item(),
                        'x2': box[2].item(), 'y2': box[3].item()}
                # label에서 카테고리 이름 추출 (예: "cauliflower(0.935)" → "cauliflower")
                cat_str = label.split('(')[0].strip()
                if cat_str in CAT_ID:
                    category_id = CAT_ID[cat_str]
                else:
                    print(f"카테고리 '{cat_str}'가 CAT_ID에 없으므로 해당 객체는 건너뜁니다.")
                    continue

                mask = masks[i]
                # (옵션) 디버그: mask의 shape 출력
                # print("mask shape:", mask.shape)

                # 중복 bbox (IoU 임계값 기반) 체크 후 annotation 생성
                if IOUcalc(registered_regions, bbox, args.iou_threshold):
                    registered_regions.append(bbox)
                    annotation = create_sub_mask_annotation(np.asarray(mask), image_id, category_id, annotation_id, is_crowd)
                    coco_annotation["annotations"].append(annotation)
                    annotation_id += 1

            # image 정보 생성 (PIL 이미지의 size는 (width, height))
            image_info = create_image_info(image_id, file, (width, height))
            coco_annotation["images"].append(image_info)

            # 결과 COCO JSON 파일 저장 (예: "image.jpg" → "image.json")
            output_json_path = os.path.join(args.target_dir, os.path.splitext(file)[0] + ".json")
            with open(output_json_path, 'w') as f:
                json.dump(coco_annotation, f, indent=4)
            print(f"COCO annotation 저장 완료: {output_json_path}")

            # (옵션) 검출 결과 시각화 – plt 창으로 띄우고자 하면 주석 해제
            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # ax.imshow(image_pil)
            # for box, label in zip(boxes_filt, pred_phrases):
            #     show_box(box.numpy(), ax, label)
            # for mask in masks:
            #     show_mask(mask, ax, random_color=True)
            # plt.axis('off')
            # plt.show()


if __name__ == "__main__":
    main()
