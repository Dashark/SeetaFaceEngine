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

#ifndef SEETA_FD_CLASSIFIER_SURF_MLP_H_
#define SEETA_FD_CLASSIFIER_SURF_MLP_H_

#include <memory>
#include <vector>

#include "classifier.h"
#include "classifier/mlp.h"
#include "feat/surf_feature_map.h"


namespace seeta {
namespace fd {

class SURFMLP : public Classifier {
 public:
  SURFMLP() : Classifier(), model_(new seeta::fd::MLP()) {}
  virtual ~SURFMLP() {}

  virtual bool Classify(fixed_t* score = nullptr, fixed_t* outputs = nullptr);

  inline virtual void SetFeatureMap(seeta::fd::FeatureMap* feat_map) {
    feat_map_ = dynamic_cast<seeta::fd::SURFFeatureMap*>(feat_map);
  }

  inline virtual seeta::fd::ClassifierType type() {
    return seeta::fd::ClassifierType::SURF_MLP;
  }

  void AddFeatureByID(int32_t feat_id);
  void AddLayer(int32_t input_dim, int32_t output_dim, const fixed_t* weights,
	  const fixed_t* bias, bool is_output = false);

  inline void SetThreshold(fixed_t thresh) { thresh_ = thresh; }

 private:
  std::vector<int32_t> feat_id_;
  std::vector<fixed_t> input_buf_;
  std::vector<fixed_t> output_buf_;

  std::shared_ptr<seeta::fd::MLP> model_;
  fixed_t thresh_;
  seeta::fd::SURFFeatureMap* feat_map_;
};

}  // namespace fd
}  // namespace seeta

#endif  // SEETA_FD_CLASSIFIER_SURF_MLP_H_
