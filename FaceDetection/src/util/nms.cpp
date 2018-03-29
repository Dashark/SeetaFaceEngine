/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include "util/nms.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace seeta {
namespace fd {

bool CompareBBox(const seeta::FaceInfo & a, const seeta::FaceInfo & b) {
  return a.score > b.score;
}

//计算交并比，合并重复的脸框
void NonMaximumSuppression(std::vector<seeta::FaceInfo>* bboxes,
  std::vector<seeta::FaceInfo>* bboxes_nms, int32_t iou_thresh) {
  bboxes_nms->clear();
  std::sort(bboxes->begin(), bboxes->end(), seeta::fd::CompareBBox);

  int32_t select_idx = 0;
  int32_t num_bbox = static_cast<int32_t>(bboxes->size());
  std::vector<int32_t> mask_merged(num_bbox, 0);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1)
      select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    bboxes_nms->push_back((*bboxes)[select_idx]);
    mask_merged[select_idx] = 1;

    seeta::Rect select_bbox = (*bboxes)[select_idx].bbox;
	int32_t area1 = select_bbox.width * select_bbox.height;
	int32_t x1 = select_bbox.x;
	int32_t y1 = select_bbox.y;
	int32_t x2 = select_bbox.x + select_bbox.width - 1;
	int32_t y2 = select_bbox.y + select_bbox.height - 1;

    select_idx++;
    for (int32_t i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1)
        continue;

      seeta::Rect & bbox_i = (*bboxes)[i].bbox;
	  int32_t x = std::max<int32_t>(x1, bbox_i.x);
	  int32_t y = std::max<int32_t>(y1, bbox_i.y);
	  int32_t w = std::min<int32_t>(x2, bbox_i.x + bbox_i.width - 1) - x + 1;
	  int32_t h = std::min<int32_t>(y2, bbox_i.y + bbox_i.height - 1) - y + 1;
      if (w <= 0 || h <= 0)
        continue;

	  int32_t area2 = bbox_i.width * bbox_i.height;
	  int32_t area_intersect = w * h;
	  int32_t area_union = area1 + area2 - area_intersect;
	  //if (fx_divx(fx_itox(area_intersect, FIXMATH_FRAC_BITS), fx_itox(area_union, FIXMATH_FRAC_BITS), FIXMATH_FRAC_BITS) > iou_thresh){
    if (area_intersect>area_union *  iou_thresh/100) {
        mask_merged[i] = 1;
        bboxes_nms->back().score += (*bboxes)[i].score;
      }
    }
  }
}

}  // namespace fd
}  // namespace seeta
