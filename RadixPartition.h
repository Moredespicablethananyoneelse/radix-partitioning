/*
 * Copyright Information Systems Group, Saarland University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * RadixPartitioning.h
 *
 * Information Systems Group
 * Saarland University
 * 2014 - 2015
 */

#ifndef RADIX_PARTITION_H_
#define RADIX_PARTITION_H_

#include <iostream>
#include "Types.h"
#include "config.h"

void radix_partition_without_buffers(Tuple* intput, Tuple* output, Index *histogram);
void radix_partition_without_buffers_v2(Tuple* intput, Tuple* output, Index *histogram);
void radix_partition_without_buffers_prefetched(Tuple *input, Tuple *output, Index *histogram);
void radix_partition_with_contiguous_buffers(Tuple* intput, Tuple* output, Index *histogram, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_prefetched(Tuple *input, Tuple *output, Index *histogram, const UInt buffered_tuples);

void radix_partition_with_contiguous_buffers_streamed_256_write(Tuple* intput, Tuple*& output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_streamed_256_write_v2(Tuple* intput, Tuple*& output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples);

void radix_partition_with_contiguous_buffers_streamed_256_write_experimental(Tuple* intput, Tuple*& output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_prefetched(Tuple * restrict input, Tuple *& restrict output, Index * restrict histogram, Index * restrict unpaddedBucketSizes, const UInt buffered_tuples);

void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_fillstate(Tuple* intput, Tuple*& output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_nothing(Tuple* intput, Tuple*& output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples);
void radix_partition_with_rw_buffers_streamed_256_write(Tuple* intput, Tuple*& output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_streamed_256_write_prefetched(Tuple* intput, Tuple*& output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_streamed_256_write_prefetched_v2(Tuple* intput, Tuple*& output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples);

void radix_partition_with_contiguous_buffers_streamed_256_write_v2_unpadded(Tuple* intput, Tuple* output, Index *histogram, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_streamed_256_write_v2_unpadded_prefetched(Tuple *input, Tuple *output, Index *histogram, const UInt buffered_tuples);

void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_nothing_unpadded(Tuple* intput, Tuple* output, Index *histogram, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_fillstate_unpadded(Tuple* intput, Tuple* output, Index *histogram, const UInt buffered_tuples);
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_unpadded(Tuple* intput, Tuple* output, Index *histogram, const UInt buffered_tuples);

#endif
