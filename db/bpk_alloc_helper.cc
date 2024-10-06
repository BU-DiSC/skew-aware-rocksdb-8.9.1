#include "bpk_alloc_helper.h"

#include <cmath>
#include <memory>
#include <unordered_map>

#include "table/block_based/block_based_table_factory.h"
#include "table/block_based/filter_policy_internal.h"

namespace ROCKSDB_NAMESPACE {

const double log_2_squared = std::pow(std::log(2), 2);

void BitsPerKeyAllocHelper::PrepareBpkAllocation(const Compaction* compaction) {
  if (ioptions_ == nullptr || vstorage_ == nullptr) return;

  if (bpk_optimization_prepared_flag_) return;

  if (ioptions_->table_factory == nullptr ||
      strcmp(ioptions_->table_factory->Name(),
             TableFactory::kBlockBasedTableName()) != 0)
    return;
  const BlockBasedTableOptions tbo =
      std::static_pointer_cast<BlockBasedTableFactory>(ioptions_->table_factory)
          ->GetBlockBasedTableOptions();
  if (tbo.filter_policy == nullptr) return;
  overall_bits_per_key_ =
      std::static_pointer_cast<const BloomLikeFilterPolicy>(tbo.filter_policy)
          ->GetBitsPerKey();
  if (overall_bits_per_key_ == 0.0) return;
  bpk_alloc_type_ = tbo.bpk_alloc_type;
  no_filter_optimize_for_level0_ = tbo.no_filter_optimize_for_level0;

  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kDefaultBpkAlloc) return;
  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kNaiveMonkeyBpkAlloc) {
    int output_level = 0;
    if (compaction != NULL) {
      output_level = compaction->output_level();
    }
    if (output_level < (int)tbo.naive_monkey_bpk_list.size()) {
      naive_monkey_bpk = tbo.naive_monkey_bpk_list[output_level];
    } else {
      naive_monkey_bpk = overall_bits_per_key_;
    }
    return;
  }
  uint64_t tmp_num_entries_in_filter_by_file = 0;
  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kDynamicMonkeyBpkAlloc ||
      (bpk_alloc_type_ == BitsPerKeyAllocationType::kWorkloadAwareBpkAlloc &&
       vstorage_->GetAccumulatedNumPointReads() == 0)) {
    bpk_alloc_type_ = BitsPerKeyAllocationType::kDynamicMonkeyBpkAlloc;

    if (compaction == nullptr &&
        !flush_flag_) {  // skip the preparation phase for flush
      return;
    }

    std::unordered_map<LevelState, uint64_t, LevelState::HashFunction>
        level2removed_num_entries;
    uint64_t added_entries_in_max_level = 0;
    num_entries_in_compaction_ = 0;
    int max_level = -1;
    if (compaction != nullptr) {
      const std::vector<CompactionInputFiles>* input_files =
          compaction->inputs();
      for (size_t i = 0; i < input_files->size(); i++) {
        int level = input_files->at(i).level;
        if (level > max_level) {
          max_level = level;
        }
        if (level != 0) {
          uint64_t num_entries = 0;
          if (level2removed_num_entries.find(LevelState(level, 0, 0)) !=
              level2removed_num_entries.end()) {
            num_entries = level2removed_num_entries[LevelState(level, 0, 0)];
          }
          uint64_t num_entries_in_bf = 0;
          for (const FileMetaData* meta : input_files->at(i).files) {
            num_entries_in_bf = meta->num_entries - meta->num_range_deletions;
            num_entries += num_entries_in_bf;
            num_entries_in_compaction_ += num_entries_in_bf;
            if (meta->bpk != -1) {
              num_bits_for_filter_to_be_removed_ +=
                  num_entries_in_bf * meta->bpk;
            } else {
              num_bits_for_filter_to_be_removed_ +=
                  num_entries_in_bf * overall_bits_per_key_;
            }
          }
          level2removed_num_entries.emplace(LevelState(level, num_entries, 0),
                                            num_entries);
        } else {
          uint64_t num_entries_in_bf = 0;
          for (const FileMetaData* meta : input_files->at(0).files) {
            num_entries_in_bf = meta->num_entries - meta->num_range_deletions;
            level2removed_num_entries.emplace(
                LevelState(0, num_entries_in_bf, meta->fd.GetNumber()),
                num_entries_in_bf);
            num_entries_in_compaction_ += num_entries_in_bf;
            if (meta->bpk != -1) {
              num_bits_for_filter_to_be_removed_ +=
                  num_entries_in_bf * meta->bpk;
            } else {
              num_bits_for_filter_to_be_removed_ +=
                  num_entries_in_bf * overall_bits_per_key_;
            }
          }
        }
      }
      if (max_level > 0) {
        added_entries_in_max_level =
            num_entries_in_compaction_ -
            level2removed_num_entries[LevelState(max_level, 0, 0)];
        level2removed_num_entries.erase(LevelState(max_level, 0, 0));
      } else if (max_level == 0 &&
                 !tbo.no_filter_optimize_for_level0) {  // intra L0 compaction
        added_entries_in_max_level = num_entries_in_compaction_;
      }
    }

    uint64_t num_entries_in_filter_by_level = 0;
    uint64_t max_fd_number = 0;
    uint64_t tmp_fd_number = 0;
    int start_level = 0;
    if (tbo.no_filter_optimize_for_level0) {
      start_level = 1;
    }
    for (int level = start_level; level < vstorage_->num_levels(); ++level) {
      num_entries_in_filter_by_level = 0;
      for (auto* file_meta : vstorage_->LevelFiles(level)) {
        tmp_num_entries_in_filter_by_file =
            file_meta->num_entries - file_meta->num_range_deletions;
        tmp_fd_number = file_meta->fd.GetNumber();
        max_fd_number = std::max(max_fd_number, tmp_fd_number);
        if (tmp_num_entries_in_filter_by_file == 0) continue;
        if (level == 0) {
          if (compaction != nullptr && !level2removed_num_entries.empty()) {
            if (level2removed_num_entries.find(LevelState(
                    0, 0, tmp_fd_number)) != level2removed_num_entries.end()) {
              // This file in level 0 is being picked to compact, so we shall
              // not consider this file as an input file in the optimization
              // procedure
              continue;
            }
          }
          level_states_pq_.push(LevelState(0, tmp_num_entries_in_filter_by_file,
                                           file_meta->fd.GetNumber()));
          temp_sum_in_bpk_optimization_ +=
              std::log(tmp_num_entries_in_filter_by_file) *
              tmp_num_entries_in_filter_by_file;
        }
        num_entries_in_filter_by_level += tmp_num_entries_in_filter_by_file;
      }
      if (max_level == level) {
        // adjust the number of entries by adding the compacted number of
        // entries from upper levels
        num_entries_in_filter_by_level += added_entries_in_max_level;
      }
      if (level != 0) {
        if (compaction != nullptr && !level2removed_num_entries.empty()) {
          if (level2removed_num_entries.find(LevelState(level, 0, 0)) !=
              level2removed_num_entries.end()) {
            num_entries_in_filter_by_level -=
                std::min(num_entries_in_filter_by_level,
                         level2removed_num_entries[LevelState(level, 0, 0)]);
          }
        }

        // skip the current level if the current level has no entries
        if (num_entries_in_filter_by_level == 0) continue;

        level_states_pq_.push(
            LevelState(level, num_entries_in_filter_by_level, 0));
        temp_sum_in_bpk_optimization_ +=
            std::log(num_entries_in_filter_by_level) *
            num_entries_in_filter_by_level;
      }
      dynamic_monkey_num_entries_ += num_entries_in_filter_by_level;
    }
    if (added_entries_in_max_level > 0 && max_level == 0 &&
        !tbo.no_filter_optimize_for_level0) {
      level_states_pq_.push(
          LevelState(0, added_entries_in_max_level, max_fd_number + 11));
      temp_sum_in_bpk_optimization_ +=
          std::log(added_entries_in_max_level) * added_entries_in_max_level;
    }
    total_bits_for_filter_ =
        dynamic_monkey_num_entries_ * overall_bits_per_key_;
    common_constant_in_bpk_optimization_ =
        -(total_bits_for_filter_ * log_2_squared +
          temp_sum_in_bpk_optimization_) /
        dynamic_monkey_num_entries_;
    std::unordered_set<size_t> levelIDs_with_bpk0_in_dynamic_monkey;
    while (!level_states_pq_.empty() &&
           std::log(level_states_pq_.top().num_entries) +
                   common_constant_in_bpk_optimization_ >
               -log_2_squared) {
      if (level_states_pq_.top().level == 0) {
        for (FileMetaData* meta : vstorage_->LevelFiles(0)) {
          if (meta->fd.GetNumber() == level_states_pq_.top().file_number) {
            meta->bpk = 0.0;
          }
        }
      } else {
        levelIDs_with_bpk0_in_dynamic_monkey.insert(
            level_states_pq_.top().level);
      }
      tmp_num_entries_in_filter_by_file = level_states_pq_.top().num_entries;
      temp_sum_in_bpk_optimization_ -=
          std::log(tmp_num_entries_in_filter_by_file) *
          tmp_num_entries_in_filter_by_file;
      dynamic_monkey_num_entries_ -= tmp_num_entries_in_filter_by_file;
      common_constant_in_bpk_optimization_ =
          -(total_bits_for_filter_ * log_2_squared +
            temp_sum_in_bpk_optimization_) /
          dynamic_monkey_num_entries_;
      level_states_pq_.pop();
    }

    vstorage_->SetLevelIDsWithEmptyBpkInDynamicMonkey(
        levelIDs_with_bpk0_in_dynamic_monkey);

    if (!level_states_pq_.empty()) {
      dynamic_monkey_bpk_num_entries_threshold_ =
          level_states_pq_.top().num_entries;
    }
  } else if (bpk_alloc_type_ ==
             BitsPerKeyAllocationType::kWorkloadAwareBpkAlloc) {
    if (compaction == nullptr &&
        !flush_flag_) {  // skip the preparation phase for flush
      return;
    }

    uint64_t num_point_reads;
    uint64_t num_existing_point_reads;
    uint64_t num_empty_point_reads;

    std::unordered_set<uint64_t> fileID_in_compaction;
    num_entries_in_compaction_ = 0;
    uint64_t num_empty_queries_in_compaction = 0;
    uint64_t num_files_in_compaction = 0;
    int max_level = -1;
    if (compaction != nullptr) {
      const std::vector<CompactionInputFiles>* input_files =
          compaction->inputs();
      std::unordered_map<int, uint64_t> level2num_point_reads;
      uint64_t tmp_num_point_reads = 0;
      uint64_t num_existing_point_reads_in_compaction = 0;
      for (size_t i = 0; i < input_files->size(); i++) {
        int level = input_files->at(i).level;
        if (level > max_level) {
          max_level = level;
        }
        num_files_in_compaction += input_files->at(i).size();
        tmp_num_point_reads = 0;
        if (level2num_point_reads.find(level) != level2num_point_reads.end()) {
          tmp_num_point_reads = level2num_point_reads[level];
        }
        uint64_t num_entries_in_bf;
        uint64_t min_num_point_reads = 0;
        for (const FileMetaData* meta : input_files->at(i).files) {
          if (i == 0) {
            min_num_point_reads =
                round(meta->stats.start_global_point_read_number *
                      vstorage_->GetAvgNumPointReadsPerLvl0File());
          }
          std::pair<uint64_t, uint64_t> est_num_point_reads =
              meta->stats.GetEstimatedNumPointReads(
                  vstorage_->GetAccumulatedNumPointReads(),
                  ioptions_->point_read_learning_rate, -1, min_num_point_reads);
          fileID_in_compaction.insert(meta->fd.GetNumber());
          num_entries_in_bf = meta->num_entries - meta->num_range_deletions;
          num_entries_in_compaction_ += num_entries_in_bf;
          if (meta->bpk != -1) {
            num_bits_for_filter_to_be_removed_ += num_entries_in_bf * meta->bpk;
          } else {
            num_bits_for_filter_to_be_removed_ +=
                num_entries_in_bf * overall_bits_per_key_;
          }
          if (est_num_point_reads.first == 0) continue;

          if (level == 0) {
            tmp_num_point_reads =
                std::max(tmp_num_point_reads, est_num_point_reads.first);
          } else {
            tmp_num_point_reads += est_num_point_reads.first;
          }
          num_existing_point_reads_in_compaction += est_num_point_reads.second;
        }
        level2num_point_reads.emplace(level, tmp_num_point_reads);
      }
      if (max_level > -1) {
        num_empty_queries_in_compaction =
            std::max(level2num_point_reads[max_level],
                     num_existing_point_reads_in_compaction) -
            num_existing_point_reads_in_compaction;
        if (max_level == 0 && tbo.no_filter_optimize_for_level0) {
          num_empty_queries_in_compaction = 0;
          num_entries_in_compaction_ = 0;
        }
      }
    }

    int start_level = 0;
    if (tbo.no_filter_optimize_for_level0) {
      start_level = 1;
    }
    for (int level = start_level; level < vstorage_->num_levels(); ++level) {
      for (auto* file_meta : vstorage_->LevelFiles(level)) {
        tmp_num_entries_in_filter_by_file =
            file_meta->num_entries - file_meta->num_range_deletions;
        if (tmp_num_entries_in_filter_by_file == 0) continue;
        if (compaction != nullptr &&
            fileID_in_compaction.find(file_meta->fd.GetNumber()) !=
                fileID_in_compaction.end())
          continue;
        uint64_t min_num_point_reads = 0;
        if (level == 0) {
          min_num_point_reads =
              round(file_meta->stats.start_global_point_read_number *
                    vstorage_->GetAvgNumPointReadsPerLvl0File());
        }
        std::pair<uint64_t, uint64_t> est_num_point_reads =
            file_meta->stats.GetEstimatedNumPointReads(
                vstorage_->GetAccumulatedNumPointReads(),
                ioptions_->point_read_learning_rate, -1, min_num_point_reads);
        num_point_reads = est_num_point_reads.first;
        num_existing_point_reads = est_num_point_reads.second;
        workload_aware_num_entries_ += tmp_num_entries_in_filter_by_file;
        if (num_point_reads == 0) continue;
        if (num_existing_point_reads < num_point_reads) {
          num_empty_point_reads = num_point_reads - num_existing_point_reads;
          total_empty_queries_ += num_empty_point_reads;
          temp_sum_in_bpk_optimization_ +=
              std::log(tmp_num_entries_in_filter_by_file * 1.0 /
                       num_empty_point_reads) *
              tmp_num_entries_in_filter_by_file;
          workload_aware_num_entries_with_empty_queries_ +=
              tmp_num_entries_in_filter_by_file;
          file_workload_state_pq_.push(
              FileWorkloadState(tmp_num_entries_in_filter_by_file,
                                num_empty_point_reads, file_meta));
        }
      }
    }

    // Adjust according to statistics in compaction
    workload_aware_num_entries_ += num_entries_in_compaction_;
    total_empty_queries_ += num_empty_queries_in_compaction;
    if (num_entries_in_compaction_ > 0 && num_empty_queries_in_compaction > 0) {
      // assuming that the number of zero-result queries distributes uniformly
      // in the newly generated files
      workload_aware_num_entries_with_empty_queries_ +=
          num_entries_in_compaction_;
      temp_sum_in_bpk_optimization_ +=
          num_entries_in_compaction_ *
          std::log(num_entries_in_compaction_ * 1.0 /
                   num_empty_queries_in_compaction);
      uint64_t avg_num_entries_in_compaction =
          num_entries_in_compaction_ * 1.0 / num_files_in_compaction;
      uint64_t avg_num_empty_queries_in_compaction =
          num_empty_queries_in_compaction * 1.0 / num_files_in_compaction;
      if (avg_num_entries_in_compaction < 1.0) {
        file_workload_state_pq_.push(
            FileWorkloadState(num_entries_in_compaction_,
                              num_empty_queries_in_compaction, nullptr));
      } else {
        for (size_t i = 0; i < num_files_in_compaction; i++) {
          file_workload_state_pq_.push(
              FileWorkloadState(avg_num_entries_in_compaction,
                                avg_num_empty_queries_in_compaction, nullptr));
        }
      }
    }

    total_bits_for_filter_ =
        workload_aware_num_entries_ * overall_bits_per_key_;
    if (total_empty_queries_ == 0 || file_workload_state_pq_.empty()) {
      bpk_optimization_prepared_flag_ = true;
      return;
    }
    temp_sum_in_bpk_optimization_ +=
        std::log(total_empty_queries_) *
        workload_aware_num_entries_with_empty_queries_;
    common_constant_in_bpk_optimization_ =
        -(total_bits_for_filter_ * log_2_squared +
          temp_sum_in_bpk_optimization_) /
        workload_aware_num_entries_with_empty_queries_;
    double weight = 0.0;
    while (
        !file_workload_state_pq_.empty() &&
        std::log(file_workload_state_pq_.top().weight * total_empty_queries_) +
                common_constant_in_bpk_optimization_ >
            -log_2_squared) {
      // if (file_workload_state_pq_.top().meta != nullptr) {
      //   file_workload_state_pq_.top().meta->bpk = 0.0;
      // }
      weight = file_workload_state_pq_.top().weight;
      temp_sum_in_bpk_optimization_ -=
          std::log(weight * total_empty_queries_) *
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

    if (!file_workload_state_pq_.empty()) {
      workload_aware_bpk_weight_threshold_ =
          file_workload_state_pq_.top().weight;
    }

    vstorage_->UpdateNumEmptyPointReads(total_empty_queries_);
  }

  bpk_optimization_prepared_flag_ = true;
}

bool BitsPerKeyAllocHelper::IfNeedAllocateBitsPerKey(
    const FileMetaData& meta, uint64_t num_entries_in_output_level,
    double* bits_per_key) {
  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kDefaultBpkAlloc)
    return false;
  assert(bits_per_key);
  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kNaiveMonkeyBpkAlloc) {
    *bits_per_key = naive_monkey_bpk;
    return true;
  }
  uint64_t num_entries = meta.num_entries - meta.num_range_deletions;
  double tmp_bits_per_key = overall_bits_per_key_;
  if (bpk_alloc_type_ == BitsPerKeyAllocationType::kDynamicMonkeyBpkAlloc ||
      (bpk_alloc_type_ == BitsPerKeyAllocationType::kWorkloadAwareBpkAlloc &&
       vstorage_->GetAccumulatedNumPointReads() == 0)) {
    if (!bpk_optimization_prepared_flag_) {
      flush_flag_ = true;
      temp_sum_in_bpk_optimization_ +=
          num_entries_in_output_level * std::log(num_entries_in_output_level);
      dynamic_monkey_num_entries_ += num_entries_in_output_level;
      level_states_pq_.push(
          LevelState(0, num_entries_in_output_level, meta.fd.GetNumber()));
      PrepareBpkAllocation();
    }

    // for bits-per-key < 1, give it 1 if it is larger than or equal to 0.5
    if (/*num_entries_in_output_level >
           dynamic_monkey_bpk_num_entries_threshold_ ||*/
        std::log(num_entries_in_output_level) +
            common_constant_in_bpk_optimization_ >
        -log_2_squared * 0.5) {
      tmp_bits_per_key = 0;
    } else {
      tmp_bits_per_key = std::max(-(std::log(num_entries_in_output_level) +
                                    common_constant_in_bpk_optimization_) /
                                      log_2_squared,
                                  1.0);
    }
  } else if (bpk_alloc_type_ ==
             BitsPerKeyAllocationType::kWorkloadAwareBpkAlloc) {
    std::pair<uint64_t, uint64_t> est_num_point_reads =
        meta.stats.GetEstimatedNumPointReads(
            vstorage_->GetAccumulatedNumPointReads(),
            ioptions_->point_read_learning_rate);

    uint64_t num_point_reads = est_num_point_reads.first;
    if (num_point_reads == 0) return false;
    uint64_t num_existing_point_reads = est_num_point_reads.second;

    if (!bpk_optimization_prepared_flag_) {
      flush_flag_ = true;
      if (num_point_reads > num_existing_point_reads) {
        total_empty_queries_ = num_point_reads - num_existing_point_reads;
      }
      temp_sum_in_bpk_optimization_ +=
          num_entries * std::log(num_entries * 1.0 / total_empty_queries_);
      workload_aware_num_entries_ += num_entries;
      workload_aware_num_entries_with_empty_queries_ += num_entries;
      PrepareBpkAllocation();
    }

    if (total_empty_queries_ == 0 || file_workload_state_pq_.empty())
      return false;

    if (num_existing_point_reads >= num_point_reads &&
        num_existing_point_reads > 0) {
      *bits_per_key = 0;
      return true;
    }

    double weight =
        num_entries * 1.0 / (num_point_reads - num_existing_point_reads);
    // for bits-per-key < 1, give it 1 if it is larger than or equal to 0.5
    if (/*weight > workload_aware_bpk_weight_threshold_ ||*/
        std::log(weight * total_empty_queries_) +
            common_constant_in_bpk_optimization_ >
        -log_2_squared * 0.5) {
      tmp_bits_per_key = 0;
    } else {
      tmp_bits_per_key = std::max(-(std::log(weight * total_empty_queries_) +
                                    common_constant_in_bpk_optimization_) /
                                      log_2_squared,
                                  1.0);
    }
  }

  uint64_t old_total_bits = vstorage_->GetCurrentTotalFilterSize() * 8.0 -
                            num_bits_for_filter_to_be_removed_;
  uint64_t old_total_entries =
      vstorage_->GetCurrentTotalNumEntries() - num_entries_in_compaction_;
  const double overused_percentage = 0.01;
  std::string stats_log = "";
  if (old_total_entries == 0 ||
      old_total_bits > old_total_entries * overall_bits_per_key_ *
                           (1 + overused_percentage)) {
    // if the old bits-per-key exceeds the overall bits-per-key (this should
    // rarely happen), we only care about if the assigned bits-per-key is larger
    // than
    if (tmp_bits_per_key > overall_bits_per_key_ * (1 + overused_percentage)) {
      return false;
    }
  } else if ((old_total_bits + tmp_bits_per_key * num_entries) * 1.0 /
                 (old_total_entries + num_entries) >
             overall_bits_per_key_ *
                 (1 +
                  overused_percentage)) {  // if the bits-per-key is overused

    tmp_bits_per_key =
        ((overall_bits_per_key_ * 1.01) * (old_total_entries + num_entries) -
         old_total_bits) /
        num_entries;
    if (tmp_bits_per_key < overall_bits_per_key_) {
      return false;
    }
  }
  *bits_per_key = tmp_bits_per_key;
  return true;
}

};  // namespace ROCKSDB_NAMESPACE
