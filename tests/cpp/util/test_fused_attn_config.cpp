/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>

#include "common/fused_attn/config_and_params.h"

namespace transformer_engine::fused_attn {

TEST(FusedAttnConfig, PackedRaggedGraphSupport) {
  EXPECT_FALSE(supports_packed_ragged_graph(90500, 90));
  EXPECT_TRUE(supports_packed_ragged_graph(90600, 90));
  EXPECT_TRUE(supports_packed_ragged_graph(91801, 100));

  // SM8x and SM120 require dense Stats/LSE and max-sequence graph dimensions,
  // even when the cuDNN runtime supports THD inputs on those architectures.
  EXPECT_FALSE(supports_packed_ragged_graph(91801, 80));
  EXPECT_FALSE(supports_packed_ragged_graph(91801, 89));
  EXPECT_FALSE(supports_packed_ragged_graph(91801, 120));
}

}  // namespace transformer_engine::fused_attn
