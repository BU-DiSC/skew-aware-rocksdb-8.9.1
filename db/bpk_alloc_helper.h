#pragma once

#include <queue>

#include "options/db_options.h"
#include "rocksdb/table.h"
#include "version_edit.h"
#include "version_set.h"

namespace ROCKSDB_NAMESPACE {

class BitsPerKeyAllocHelper {
 public:
  explicit BitsPerKeyAllocHelper() { bpk_optimization_prepared_flag_ = false; }

  // Prepare bpk optimizations (calculate the temporary sum and the common
  // constant that are used in next-step bpk reassignment)
  void PrepareBpkAllocation(const ImmutableOptions* immutable_options,
                            const VersionStorageInfo* vstorage);
  // Check if we need to allocate bits-per-key (newly allocated bpk is stored in
  // *bitis_per_key)
  bool IfNeedAllocateBitsPerKey(const FileMetaData& meta,
                                uint64_t num_entries_in_output_level,
                                double* bits_per_key) const;

  BitsPerKeyAllocationType bpk_alloc_type_ =
      BitsPerKeyAllocationType::kDefaultBpkAlloc;
  bool bpk_optimization_prepared_flag_;
  double workload_aware_bpk_weight_threshold_ = 0.0;
  uint64_t monkey_bpk_num_entries_threshold_ = 0;
  uint64_t workload_aware_num_entries_with_empty_queries_ = 0;
  double temp_sum_in_bpk_optimization_ = 0;
  double common_constant_in_bpk_optimization_ = 0;
  uint64_t total_empty_queries_ = 0;
  double total_bits_for_filter_ = 0.0;
  struct LevelState {
    int level;
    uint64_t num_entries;
    uint64_t file_number;
    LevelState(int _level, uint64_t _num_entries, uint64_t _file_number) {
      level = _level;
      num_entries = _num_entries;
      file_number = _file_number;
    }
    bool operator<(const LevelState& tmp) const {
      if (num_entries < tmp.num_entries) return true;
      if (level > tmp.level) return true;
      return file_number < tmp.file_number;
    }
  };
  // used by kMonkey
  std::priority_queue<LevelState> level_states_pq_;

  struct FileWorkloadState {
    uint64_t num_entries;
    uint64_t num_empty_point_reads;
    double weight;
    FileWorkloadState(uint64_t _num_entries, uint64_t _num_empty_point_reads) {
      num_entries = _num_entries;
      num_empty_point_reads = _num_empty_point_reads;
      if (num_empty_point_reads != 0) {
        weight = num_entries * 1.0 / num_empty_point_reads;
      }
    }
    bool operator<(const FileWorkloadState& tmp) const {
      if (num_empty_point_reads == 0) return false;
      if (tmp.num_empty_point_reads == 0) return true;
      return weight < tmp.weight;
    }
  };
  // used by kWorkloadAware
  std::priority_queue<FileWorkloadState> file_workload_state_pq_;
};

};  // namespace ROCKSDB_NAMESPACE