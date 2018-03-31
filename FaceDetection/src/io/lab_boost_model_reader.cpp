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

#include "io/lab_boost_model_reader.h"

#include <vector>

namespace seeta {
namespace fd {

bool LABBoostModelReader::Read(std::istream* input,
    seeta::fd::Classifier* model) {
  bool is_read;
  seeta::fd::LABBoostedClassifier* lab_boosted_classifier =
    dynamic_cast<seeta::fd::LABBoostedClassifier*>(model);

  input->read(reinterpret_cast<char*>(&num_base_classifer_), sizeof(int32_t));
  input->read(reinterpret_cast<char*>(&num_bin_), sizeof(int32_t));
  bin_buffer_ = new float[num_bin_ + 1];
  classifier_buffer_ = new float[num_base_classifer_];

  is_read = (!input->fail()) && num_base_classifer_ > 0 && num_bin_ > 0 &&
    ReadFeatureParam(input, lab_boosted_classifier) &&
    ReadBaseClassifierParam(input, lab_boosted_classifier);

  return is_read;
}

bool LABBoostModelReader::ReadFeatureParam(std::istream* input,
    seeta::fd::LABBoostedClassifier* model) {
  int32_t x;
  int32_t y;
  for (int32_t i = 0; i < num_base_classifer_; i++) {
    input->read(reinterpret_cast<char*>(&x), sizeof(int32_t));
    input->read(reinterpret_cast<char*>(&y), sizeof(int32_t));
    model->AddFeature(x, y);
  }

  return !input->fail();
}

bool LABBoostModelReader::ReadBaseClassifierParam(std::istream* input,
    seeta::fd::LABBoostedClassifier* model) {
  //std::vector<float> thresh;
  //thresh.resize(num_base_classifer_);
  input->read(reinterpret_cast<char*>(classifier_buffer_),
    sizeof(float)* num_base_classifer_);

  //开始转换
  fixed_t* classifier_buffer_fx = new fixed_t[num_base_classifer_];
  for (int32_t a=0; a < num_base_classifer_; a++){
	  classifier_buffer_fx[a] = fx_ftox(classifier_buffer_[a], FIXMATH_FRAC_BITS);
  }
  //结束转换


  int32_t weight_len = sizeof(float)* (num_bin_ + 1);
  //std::vector<float> weights;
  //weights.resize(num_bin_ + 1);
  for (int32_t i = 0; i < num_base_classifer_; i++) {
    input->read(reinterpret_cast<char*>(bin_buffer_), weight_len);
	//开始转换
	fixed_t* bin_buffer_fx = new fixed_t[num_bin_ + 1];
	for (int32_t a; a < num_bin_ + 1; a++){
		bin_buffer_fx[a] = fx_ftox(bin_buffer_[a], FIXMATH_FRAC_BITS);
	}
	//结束转换
	model->AddBaseClassifier(bin_buffer_fx, num_bin_, classifier_buffer_fx[i]);
  }

  return !input->fail();
}

}  // namespace fd
}  // namespace seeta
