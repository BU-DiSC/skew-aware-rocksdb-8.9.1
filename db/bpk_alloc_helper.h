#pragma once

#include <limits>
#include <queue>

#include "compaction/compaction.h"
#include "options/db_options.h"
#include "rocksdb/table.h"
#include "version_edit.h"
#include "version_set.h"

namespace ROCKSDB_NAMESPACE {

class BitsPerKeyAllocHelper {
 public:
  explicit BitsPerKeyAllocHelper(const ImmutableOptions* immutable_options,
                                 const VersionStorageInfo* vstorage)
      : ioptions_(immutable_options), vstorage_(vstorage) {
    mnemosyne_bpk_optimization_prepared_flag_ = false;
    mnemosyne_plus_bpk_optimization_prepared_flag_ = false;
    num_bits_for_filter_to_be_removed_ = 0;
    num_entries_in_compaction_ = 0;
  }

  // Prepare bpk optimizations (calculate the temporary sum and the common
  // constant that are used in next-step bpk reassignment)
  void PrepareBpkAllocation(const Compaction* compaction = nullptr);
  // Check if we need to allocate bits-per-key (newly allocated bpk is stored in
  // *bitis_per_key)
  bool IfNeedAllocateBitsPerKey(const FileMetaData& meta,
                                uint64_t num_entries_in_output_level,
                                double* bits_per_key,
                                bool is_last_level = false,
                                const Compaction* compaction = nullptr);
  // pre-process the statistics to prepare Mnemosyne-based bits-per-key
  // allocation
  void PrepareForMnemosyne(const Compaction* compaction);
  // pre-process the statistics to prepare Mnemosyne-plus-based bits-per-key
  // allocation
  void PrepareForMnemosynePlus(const Compaction* compaction);

  double avg_curr_bits_per_key = 0.0;
  uint64_t agg_filter_size_ = 0;

  const ImmutableOptions* ioptions_;
  const VersionStorageInfo* vstorage_;
  BitsPerKeyAllocationType bpk_alloc_type_ =
      BitsPerKeyAllocationType::kDefaultBpkAlloc;
  bool flush_flag_ = false;
  double naive_monkey_bpk = 0.0;
  bool mnemosyne_bpk_optimization_prepared_flag_ = false;
  bool mnemosyne_plus_bpk_optimization_prepared_flag_ = false;
  bool no_filter_optimize_for_level0_ = false;
  double mnemosyne_plus_bpk_weight_threshold_ =
      std::numeric_limits<double>::max();
  uint64_t mnemosyne_bpk_num_entries_threshold_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t mnemosyne_num_entries_ = 0;
  uint64_t mnemosyne_plus_num_entries_ = 0;
  uint64_t mnemosyne_plus_num_entries_with_empty_queries_ = 0;
  double mnemosyne_plus_temp_sum_in_bpk_optimization_ = 0;
  double mnemosyne_plus_common_constant_in_bpk_optimization_ = 0;
  uint64_t mnemosyne_plus_total_empty_queries_ = 0;
  double total_bits_for_filter_ = 0.0;

  double overall_bits_per_key_ = 0.0;
  uint64_t num_entries_in_compaction_ = 0;
  uint64_t num_bits_for_filter_to_be_removed_ = 0;

  double mnemosyne_temp_sum_in_bpk_optimization_ = 0;
  double mnemosyne_common_constant_in_bpk_optimization_ = 0;
  double max_bits_per_key_ = 100;
  struct LevelState {
    int level;
    uint64_t num_entries;
    uint64_t file_number;
    uint64_t agg_filter_size;
    LevelState(int _level, uint64_t _num_entries, uint64_t _file_number,
               uint64_t _agg_filter_size = 0) {
      level = _level;
      num_entries = _num_entries;
      file_number = _file_number;
      agg_filter_size = _agg_filter_size;
    }
    bool operator<(const LevelState& tmp) const {
      if (num_entries < tmp.num_entries) return true;
      if (level > tmp.level) return true;
      return file_number < tmp.file_number;
    }
    bool operator==(const LevelState& tmp) const {
      return level == tmp.level && file_number == tmp.file_number;
    }
    bool operator!=(const LevelState& tmp) const {
      return level != tmp.level || file_number != tmp.file_number;
    }
    struct HashFunction {
      size_t operator()(const LevelState& level_state) const {
        size_t levelHash = std::hash<int>()(level_state.level);
        size_t filenumberHash =
            (std::hash<uint64_t>()(level_state.file_number) << 17) |
            (std::hash<uint64_t>()(level_state.file_number) >> 47);
        return levelHash ^ filenumberHash;
      }
    };
  };
  // used by kMnemosyne
  std::priority_queue<LevelState> level_states_pq_;

  struct FileWorkloadState {
    double num_entries;
    double num_empty_point_reads;
    double weight;
    const FileMetaData* meta;
    FileWorkloadState(double _num_entries, double _num_empty_point_reads,
                      const FileMetaData* _meta)
        : num_entries(_num_entries),
          num_empty_point_reads(_num_empty_point_reads),
          meta(_meta) {
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
  // used by kMnemosynePlus
  std::priority_queue<FileWorkloadState> file_workload_state_pq_;
};

using LevelState = rocksdb::BitsPerKeyAllocHelper::LevelState;
};  // namespace ROCKSDB_NAMESPACE
