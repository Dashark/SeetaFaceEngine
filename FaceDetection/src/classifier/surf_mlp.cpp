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

#include "classifier/surf_mlp.h"

#include<stdio.h>
#include <string>

namespace seeta {
namespace fd {

bool SURFMLP::Classify(float* score, float* outputs) {
  float* dest = input_buf_.data();

  //��ʼת��
  std::vector<fixed_t> input_buf_fx;
  input_buf_fx.resize(model_->GetInputDim());
  for (int32_t i = 0; i < model_->GetInputDim(); i++){
	  input_buf_fx[i] = fx_ftox(input_buf_[i], FIXMATH_FRAC_BITS);
  }
  fixed_t* dest_fx = input_buf_fx.data();
  //����ת��

  for (size_t i = 0; i < feat_id_.size(); i++) {
	  feat_map_->GetFeatureVector(feat_id_[i] - 1, dest_fx);
	  dest_fx += feat_map_->GetFeatureVectorDim(feat_id_[i]);
  }



  //��ʼת��
  std::vector<fixed_t> output_buf_fx;
  output_buf_fx.resize(model_->GetOutputDim());
  //����ת��
  

  output_buf_.resize(model_->GetOutputDim());
  model_->Compute(input_buf_fx.data(), output_buf_fx.data());

  //��ʼת��
  for (int32_t i = 0; i < input_buf_.size(); i++){
	  input_buf_[i] = fx_xtof(input_buf_fx[i], FIXMATH_FRAC_BITS);
  }
  for (int32_t i = 0; i < output_buf_.size(); i++){
	  output_buf_[i] = fx_xtof(output_buf_fx[i], FIXMATH_FRAC_BITS);
  }
  //����ת��


  if (score != nullptr)
    *score = output_buf_[0];
  if (outputs != nullptr) {
    std::memcpy(outputs, output_buf_.data(),
      model_->GetOutputDim() * sizeof(float));
  }

  return (output_buf_[0] > fx_xtof(thresh_, FIXMATH_FRAC_BITS));
}

void SURFMLP::AddFeatureByID(int32_t feat_id) {
  feat_id_.push_back(feat_id);
}

void SURFMLP::AddLayer(int32_t input_dim, int32_t output_dim,
	const fixed_t* weights, const fixed_t* bias, bool is_output) {
  if (model_->GetLayerNum() == 0)
    input_buf_.resize(input_dim);
  model_->AddLayer(input_dim, output_dim, weights, bias, is_output);
}

}  // namespace fd
}  // namespace seeta
