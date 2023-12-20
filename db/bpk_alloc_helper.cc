#include "bpk_alloc_helper.h"

#include <memory>

#include "table/block_based/block_based_table_factory.h"
#include "table/block_based/filter_policy_internal.h"

namespace ROCKSDB_NAMESPACE {

const double log_2_squared = std::pow(std::log(2), 2);

void BitsPerKeyAllocHelper::PrepareBpkAllocation(
    const ImmutableOptions* immutable_options,
    const VersionStorageInfo* vstorage) {
  if (immutable_options == nullptr || vstorage == nullptr) return;

  if (bpk_optimization_prepared_flag_) return;

  if (immutable_options->table_factory == nullptr ||
      strcmp(immutable_options->table_factory->Name(),
             TableFactory::kBlockBasedTableName()) != 0)
    return;

  const BlockBasedTableOptions tbo =
      std::static_pointer_cast<BlockBasedTableFactory>(
          immutable_options->table_factory)
          ->GetBlockBasedTableOptions();
  if (tbo.filter_policy == nullptr) return;
  double overall_bits_per_key =
      std::static_pointer_cast<const BloomLikeFilterPolicy>(tbo.filter_policy)
          ->GetBitsPerKey();
  if (overall_bits_per_key == 0.0) return;
  bpk_alloc_type_ = tbo.bpk_alloc_type;

  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kDefaultBpkAlloc) return;
  uint64_t num_entries_in_filter = 0;
  uint64_t tmp_num_entries_in_filter_by_file = 0;
  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kMonkeyBpkAlloc) {
    uint64_t num_entries_in_filter_by_level = 0;
    for (int level = 0; level < vstorage->num_levels(); ++level) {
      num_entries_in_filter_by_level = 0;
      for (auto* file_meta : vstorage->LevelFiles(level)) {
        tmp_num_entries_in_filter_by_file =
            file_meta->num_entries - file_meta->num_range_deletions;
        if (tmp_num_entries_in_filter_by_file == 0) continue;
        if (level == 0) {
          level_states_pq_.push(LevelState(0, tmp_num_entries_in_filter_by_file,
                                           file_meta->fd.GetNumber()));
          temp_sum_in_bpk_optimization_ +=
              std::log(tmp_num_entries_in_filter_by_file) *
              tmp_num_entries_in_filter_by_file;
        }
        num_entries_in_filter_by_level += tmp_num_entries_in_filter_by_file;
      }
      num_entries_in_filter += num_entries_in_filter_by_level;
      if (level != 0) {
        level_states_pq_.push(
            LevelState(level, num_entries_in_filter_by_level, 0));
        temp_sum_in_bpk_optimization_ +=
            std::log(num_entries_in_filter_by_level) *
            num_entries_in_filter_by_level;
      }
    }
    total_bits_for_filter_ = num_entries_in_filter * overall_bits_per_key;
    common_constant_in_bpk_optimization_ =
        -(total_bits_for_filter_ * log_2_squared +
          temp_sum_in_bpk_optimization_) /
        num_entries_in_filter;
    while (!level_states_pq_.empty() &&
           std::log(level_states_pq_.top().num_entries) +
                   common_constant_in_bpk_optimization_ >
               std::exp(-log_2_squared)) {
      tmp_num_entries_in_filter_by_file = level_states_pq_.top().num_entries;
      temp_sum_in_bpk_optimization_ -=
          std::log(tmp_num_entries_in_filter_by_file) *
          tmp_num_entries_in_filter_by_file;
      num_entries_in_filter -= tmp_num_entries_in_filter_by_file;
      common_constant_in_bpk_optimization_ =
          -(total_bits_for_filter_ * log_2_squared +
            temp_sum_in_bpk_optimization_) /
          num_entries_in_filter;
      level_states_pq_.pop();
    }
    if (!level_states_pq_.empty())
      monkey_bpk_num_entries_threshold_ = level_states_pq_.top().num_entries;
  } else if (bpk_alloc_type_ ==
             BitsPerKeyAllocationType::kWorkloadAwareBpkAlloc) {
    uint64_t num_point_reads;
    uint64_t num_existing_point_reads;
    uint64_t num_empty_point_reads;
    for (int level = 0; level < vstorage->num_levels(); ++level) {
      for (auto* file_meta : vstorage->LevelFiles(level)) {
        tmp_num_entries_in_filter_by_file =
            file_meta->num_entries - file_meta->num_range_deletions;
        if (tmp_num_entries_in_filter_by_file == 0) continue;
        num_point_reads =
            file_meta->stats.num_point_reads.load(std::memory_order_relaxed);
        num_existing_point_reads =
            file_meta->stats.num_existing_point_reads.load(
                std::memory_order_relaxed);
        if (num_existing_point_reads < num_point_reads) {
          num_empty_point_reads = num_point_reads - num_existing_point_reads;
          total_empty_queries_ += num_empty_point_reads;
          temp_sum_in_bpk_optimization_ +=
              std::log(tmp_num_entries_in_filter_by_file * 1.0 /
                       num_empty_point_reads) *
              tmp_num_entries_in_filter_by_file;
          workload_aware_num_entries_with_empty_queries_ +=
              tmp_num_entries_in_filter_by_file;
          file_workload_state_pq_.push(FileWorkloadState(
              tmp_num_entries_in_filter_by_file, num_empty_point_reads));
        }
        num_entries_in_filter += tmp_num_entries_in_filter_by_file;
      }
    }
    temp_sum_in_bpk_optimization_ +=
        std::log(total_empty_queries_) *
        workload_aware_num_entries_with_empty_queries_;
    total_bits_for_filter_ = num_entries_in_filter * overall_bits_per_key;
    common_constant_in_bpk_optimization_ =
        -(total_bits_for_filter_ * log_2_squared +
          temp_sum_in_bpk_optimization_) /
        workload_aware_num_entries_with_empty_queries_;
    double weight = 0.0;
    while (!file_workload_state_pq_.empty() &&
           std::log(file_workload_state_pq_.top().weight) *
                       workload_aware_num_entries_with_empty_queries_ +
                   common_constant_in_bpk_optimization_ >
               std::exp(-log_2_squared)) {
      weight = file_workload_state_pq_.top().weight;
      temp_sum_in_bpk_optimization_ -=
          std::log(weight * workload_aware_num_entries_with_empty_queries_) *
          file_workload_state_pq_.top().num_entries;
      workload_aware_num_entries_with_empty_queries_ -=
          file_workload_state_pq_.top().num_entries;
      temp_sum_in_bpk_optimization_ +=
          workload_aware_num_entries_with_empty_queries_ *
          std::log((total_empty_queries_ -
                    file_workload_state_pq_.top().num_empty_point_reads) *
                   1.0 / total_empty_queries_);
      total_empty_queries_ -=
          file_workload_state_pq_.top().num_empty_point_reads;
      common_constant_in_bpk_optimization_ =
          -(total_bits_for_filter_ * log_2_squared +
            temp_sum_in_bpk_optimization_) /
          workload_aware_num_entries_with_empty_queries_;
      file_workload_state_pq_.pop();
    }
    if (!file_workload_state_pq_.empty())
      workload_aware_bpk_weight_threshold_ =
          file_workload_state_pq_.top().weight;
  }

  bpk_optimization_prepared_flag_ = true;
}

bool BitsPerKeyAllocHelper::IfNeedAllocateBitsPerKey(
    const FileMetaData& meta, uint64_t num_entries_in_output_level,
    double* bits_per_key) const {
  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kDefaultBpkAlloc)
    return false;
  assert(bits_per_key);

  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kMonkeyBpkAlloc) {
    if (num_entries_in_output_level > monkey_bpk_num_entries_threshold_) {
      *bits_per_key = 0;
    }

    *bits_per_key = -(std::log(num_entries_in_output_level) +
                      common_constant_in_bpk_optimization_) /
                    log_2_squared;

  } else if (bpk_alloc_type_ ==
             BitsPerKeyAllocationType::kWorkloadAwareBpkAlloc) {
    uint64_t num_entries = meta.num_entries - meta.num_range_deletions;
    uint64_t num_point_reads =
        meta.stats.num_point_reads.load(std::memory_order_relaxed);
    uint64_t num_existing_point_reads =
        meta.stats.num_existing_point_reads.load(std::memory_order_relaxed);

    if (num_existing_point_reads >= num_point_reads) {
      *bits_per_key = 0;
    }

    double weight = num_entries * 1.0 /
                    (num_point_reads - num_existing_point_reads) *
                    total_empty_queries_;
    if (weight >= workload_aware_bpk_weight_threshold_) {
      *bits_per_key = 0;
    }

    *bits_per_key =
        -(std::log(num_entries * (num_point_reads - num_existing_point_reads) *
                   1.0 / total_empty_queries_) +
          common_constant_in_bpk_optimization_) /
        log_2_squared;
  }
  return true;
}

};  // namespace ROCKSDB_NAMESPACE
