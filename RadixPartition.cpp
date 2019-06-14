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
 * RadixPartitioning.cpp
 *
 * Information Systems Group
 * Saarland University
 * 2014 - 2015
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Types.h"
#include "Macros.h"
#include "RadixPartition.h"
#include "emmintrin.h"
#include "immintrin.h"
#include "smmintrin.h"

#if defined(UNBUFFERED) || defined(UNBUFFERED_4KB_PAGE) || defined(UNBUFFERED_2MB_PAGE)
double time_difference(struct timeval& first, struct timeval& second);
#include <sys/time.h>
void radix_partition_without_buffers(Tuple *input, Tuple *output, Index *histogram) {
	struct timeval init, build, acc, final;

	gettimeofday(&init, NULL);

	constexpr UInt shift = 32 - log2partitions();
	__attribute__((aligned(64))) Index *final_buckets;

	posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	gettimeofday(&build, NULL);

	for(Index i = 1; i < N_PARTITIONS; ++i){
		final_buckets[i] += final_buckets[i - 1];
	}

	memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

	gettimeofday(&acc, NULL);

	__attribute__((aligned(64))) Index bucket_num = 0;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		output[final_buckets[bucket_num] - 1] = input[j];
		--final_buckets[bucket_num];
	}

	free(final_buckets);

	gettimeofday(&final, NULL);

	std::cout << time_difference(init, build) << std::endl << time_difference(build, acc) << std::endl << time_difference(acc, final) << std::endl;
}
#endif

#ifdef UNBUFFERED_V2
void radix_partition_without_buffers_v2(Tuple *input, Tuple *output, Index *histogram) {
	constexpr UInt shift = shift_distance();

	__attribute__((aligned(64))) Index *final_buckets;
	posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

	__attribute__((aligned(64))) Index *targets;
	posix_memalign((void**)&targets, 64, N_PARTITIONS * sizeof(Index));

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	Index offset = 0;
	for(Index i = 0; i < N_PARTITIONS; ++i) {
		targets[i] = offset;
		offset += final_buckets[i];
	}

	// to make sure the check outside works
	for(Index i = 1; i < N_PARTITIONS; ++i){
		final_buckets[i] += final_buckets[i - 1];
	}
	memcpy(histogram, targets, N_PARTITIONS * sizeof(Index));

	__attribute__((aligned(64))) Index bucket_num = 0;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		output[targets[bucket_num]++] = input[j];
	}

	free(final_buckets);
}
#endif

#ifdef UNBUFFERED_PREFETCHED
void radix_partition_without_buffers_prefetched(Tuple *input, Tuple *output, Index *histogram) {
	constexpr UInt shift = shift_distance();
	__attribute__((aligned(64))) Index *final_buckets;

	posix_memalign((void**)&final_buckets, 64, (N_PARTITIONS + 1) * sizeof(Index));
	memset(final_buckets, 0, (N_PARTITIONS + 1) * sizeof(Index));

	final_buckets++;
	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	final_buckets--;
	for(Index i = 1; i <= N_PARTITIONS; ++i){
		final_buckets[i] += final_buckets[i - 1];
	}

	memcpy(histogram, final_buckets + 1, N_PARTITIONS * sizeof(Index));

	__attribute__((aligned(64))) Index bucket_num = 0;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		output[final_buckets[bucket_num]++] = input[j];
		__builtin_prefetch(output + final_buckets[bucket_num], 1, 0);
	}

	free(final_buckets);
}
#endif

#ifdef CONTIGUOUS_BUFFERS
void radix_partition_with_contiguous_buffers(Tuple *input, Tuple *output, Index *histogram, const UInt buffered_tuples) {
	constexpr UInt shift = shift_distance();
	__attribute__((aligned(64))) Index *final_buckets;
	__attribute__((aligned(64))) Tuple *buffers;
	__attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];

	posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));
	posix_memalign((void**)&buffers, 64, N_PARTITIONS * buffered_tuples * sizeof(Tuple));
	memset(buffer_counters, 0, N_PARTITIONS * sizeof(UInt));

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	for(Index i = 1; i < N_PARTITIONS; ++i){
		final_buckets[i] += final_buckets[i - 1];
	}

	memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

	__attribute__((aligned(64))) Index bucket_num = 0;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		Index offset = bucket_num * buffered_tuples;
		buffers[offset + buffer_counters[bucket_num]] = input[j];
		buffer_counters[bucket_num]++;
		if (buffer_counters[bucket_num] == buffered_tuples) {
			final_buckets[bucket_num] -= buffered_tuples;
			memcpy(output + final_buckets[bucket_num], buffers + offset, sizeof(Tuple) * buffered_tuples);
			buffer_counters[bucket_num] = 0;
		}
	}

	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index offset = i * buffered_tuples;
		for (UInt b = 0; b < buffer_counters[i]; b++) {
			output[final_buckets[i] - 1] = buffers[offset + b];
			--final_buckets[i];
		}
	}

	free(final_buckets);
	free(buffers);
}
#endif

#ifdef CONTIGUOUS_BUFFERS_PREFETCHED
void radix_partition_with_contiguous_buffers_prefetched(Tuple *input, Tuple *output, Index *histogram, const UInt buffered_tuples) {
    constexpr UInt shift = shift_distance();
    __attribute__((aligned(64))) Index *final_buckets;
    __attribute__((aligned(64))) Tuple *buffers;
    __attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];

    posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
    memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));
    posix_memalign((void**)&buffers, 64, N_PARTITIONS * buffered_tuples * sizeof(Tuple));
    memset(buffer_counters, 0, N_PARTITIONS * sizeof(UInt));

    for(Index j = 0; j < ELEMS; ++j){
        ++final_buckets[GET_BUCKET(input[j].value, shift)];
    }

    for(Index i = 1; i < N_PARTITIONS; ++i){
        final_buckets[i] += final_buckets[i - 1];
    }

    memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

    __attribute__((aligned(64))) Index bucket_num = 0;
    for(Index j = 0; j < ELEMS; ++j){
        bucket_num = GET_BUCKET(input[j].value, shift);
        Index offset = bucket_num * buffered_tuples;
        buffers[offset + buffer_counters[bucket_num]] = input[j];
        buffer_counters[bucket_num]++;
        __builtin_prefetch(buffers + offset + buffer_counters[bucket_num], 1, 0);
        if (buffer_counters[bucket_num] == buffered_tuples) {
            final_buckets[bucket_num] -= buffered_tuples;
            memcpy(output + final_buckets[bucket_num], buffers + offset, sizeof(Tuple) * buffered_tuples);
            buffer_counters[bucket_num] = 0;
        }
    }

    for (Index i = 0; i < N_PARTITIONS; i++) {
        Index offset = i * buffered_tuples;
        for (UInt b = 0; b < buffer_counters[i]; b++) {
            output[final_buckets[i] - 1] = buffers[offset + b];
            --final_buckets[i];
        }
    }

    free(final_buckets);
    free(buffers);
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_PREFETCHED
void radix_partition_with_contiguous_buffers_streamed_256_write_prefetched(Tuple *input, Tuple *&output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples) {
	constexpr UInt shift = 32 - log2partitions();
	__attribute__((aligned(64))) Index *final_buckets;
	__attribute__((aligned(64))) Tuple *buffers;
	__attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];

	posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));
	posix_memalign((void**)&buffers, 64, N_PARTITIONS * buffered_tuples * sizeof(Tuple));
	memset(buffer_counters, 0, N_PARTITIONS * sizeof(UInt));

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	memcpy(unpaddedBucketSizes, final_buckets, N_PARTITIONS * sizeof(Index));

	// we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
	// if this is not the case for a partition, we add padding elements
	size_t rest;
	for(Index i = 0; i < N_PARTITIONS; ++i){
		if((rest = (final_buckets[i] % 4)) != 0) {
			final_buckets[i] += 4 - rest;
		}
	}

	for(Index i = 1; i < N_PARTITIONS; ++i){
		final_buckets[i] += final_buckets[i - 1];
	}

	memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

	// allocate output according to histogram
	posix_memalign((void**)&output, 64, final_buckets[N_PARTITIONS - 1] * sizeof(Tuple));

	//_mm_stream_si64 (__int64* mem_addr, __int64 a)
	__attribute__((aligned(64))) Index bucket_num = 0;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		Index offset = bucket_num * buffered_tuples;
		buffers[offset + buffer_counters[bucket_num]++] = input[j];
		__builtin_prefetch(buffers + offset + buffer_counters[bucket_num], 1, 0);
		if (buffer_counters[bucket_num] == buffered_tuples) {
			for(UInt b = 0; b < buffered_tuples; b += 1) {
				__builtin_prefetch(buffers + offset + b, 1, 0);
			}
			final_buckets[bucket_num] -= buffered_tuples;
			for(UInt b = 0; b < buffered_tuples; b += 4) {
				_mm256_stream_si256(reinterpret_cast<__m256i*>(output + final_buckets[bucket_num]), _mm256_load_si256((reinterpret_cast<__m256i*>(buffers + offset + b))));
				final_buckets[bucket_num] += 4;
			}
			final_buckets[bucket_num] -= buffered_tuples;
			buffer_counters[bucket_num] = 0;
		}
	}

	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index offset = i * buffered_tuples;
		for (UInt b = 0; b < buffer_counters[i]; b++) {
			output[final_buckets[i] - 1] = buffers[offset + b];
			--final_buckets[i];
		}
	}

	free(final_buckets);
	free(buffers);
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_PREFETCHED_V2
void radix_partition_with_contiguous_buffers_streamed_256_write_prefetched_v2(Tuple *input, Tuple *&output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples) {
	__attribute__((aligned(64))) Index *final_buckets;
	__attribute__((aligned(64))) Tuple *buffers;
	__attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];
	constexpr UInt shift = shift_distance();

	posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));
	posix_memalign((void**)&buffers, 64, N_PARTITIONS * buffered_tuples * sizeof(Tuple));
	memset(buffer_counters, 0, N_PARTITIONS * sizeof(UInt));

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	memcpy(unpaddedBucketSizes, final_buckets, N_PARTITIONS * sizeof(Index));

	// we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
	// if this is not the case for a partition, we add padding elements
	size_t rest;
	for(Index i = 0; i < N_PARTITIONS; ++i){
		if((rest = (final_buckets[i] % TUPLES_PER_CACHELINE)) != 0) {
			final_buckets[i] += TUPLES_PER_CACHELINE - rest;
		}
	}

	for(Index i = 1; i < N_PARTITIONS; ++i){
		final_buckets[i] += final_buckets[i - 1];
	}

	memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

	// allocate output according to histogram
	posix_memalign((void**)&output, 64, final_buckets[N_PARTITIONS - 1] * sizeof(Tuple));

	//_mm_stream_si64 (__int64* mem_addr, __int64 a)
	__attribute__((aligned(64))) Index bucket_num = 0;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		Index offset = bucket_num * buffered_tuples;
		buffers[offset + buffer_counters[bucket_num]++] = input[j];
		__builtin_prefetch(buffers + offset + buffer_counters[bucket_num], 1, 0);
		if (buffer_counters[bucket_num] == buffered_tuples) {
			for(UInt b = 0; b < buffered_tuples; b += 1) {
				__builtin_prefetch(buffers + offset + b, 1, 0);
			}
			final_buckets[bucket_num] -= buffered_tuples;
			for(UInt b = 0; b < buffered_tuples; b += STREAM_UNIT) {
				_mm256_stream_si256(reinterpret_cast<__m256i*>(output + final_buckets[bucket_num]), _mm256_load_si256((reinterpret_cast<__m256i*>(buffers + offset + b))));
				final_buckets[bucket_num] += STREAM_UNIT;
			}
			final_buckets[bucket_num] -= buffered_tuples;
			buffer_counters[bucket_num] = 0;
		}
	}

	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index offset = i * buffered_tuples;
		for (UInt b = 0; b < buffer_counters[i]; b++) {
			output[final_buckets[i] - 1] = buffers[offset + b];
			--final_buckets[i];
		}
	}

	free(final_buckets);
	free(buffers);
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM
void radix_partition_with_contiguous_buffers_streamed_256_write(Tuple *input, Tuple *&output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples) {
	constexpr UInt shift = 32 - log2partitions();
	__attribute__((aligned(64))) Index *final_buckets;
	__attribute__((aligned(64))) Tuple *buffers;
	__attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];

	posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));
	posix_memalign((void**)&buffers, 64, N_PARTITIONS * buffered_tuples * sizeof(Tuple));
	memset(buffer_counters, 0, N_PARTITIONS * sizeof(UInt));

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	memcpy(unpaddedBucketSizes, final_buckets, N_PARTITIONS * sizeof(Index));

	// we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
	// if this is not the case for a partition, we add padding elements
	size_t rest;
	for(Index i = 0; i < N_PARTITIONS; ++i){
		if((rest = (final_buckets[i] % 4)) != 0) {
			final_buckets[i] += 4 - rest;
		}
	}

	for(Index i = 1; i < N_PARTITIONS; ++i){
		final_buckets[i] += final_buckets[i - 1];
	}

	memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

	// allocate output according to histogram
	posix_memalign((void**)&output, 64, final_buckets[N_PARTITIONS - 1] * sizeof(Tuple));

	//_mm_stream_si64 (__int64* mem_addr, __int64 a)
	__attribute__((aligned(64))) Index bucket_num = 0;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		Index offset = bucket_num * buffered_tuples;
		buffers[offset + buffer_counters[bucket_num]++] = input[j];
		if (buffer_counters[bucket_num] == buffered_tuples) {
			final_buckets[bucket_num] -= buffered_tuples;
			for(UInt b = 0; b < buffered_tuples; b += 4) {
				_mm256_stream_si256(reinterpret_cast<__m256i*>(output + final_buckets[bucket_num]), _mm256_load_si256((reinterpret_cast<__m256i*>(buffers + offset + b))));
				final_buckets[bucket_num] += 4;
			}
			final_buckets[bucket_num] -= buffered_tuples;
			buffer_counters[bucket_num] = 0;
		}
	}

	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index offset = i * buffered_tuples;
		for (UInt b = 0; b < buffer_counters[i]; b++) {
			output[final_buckets[i] - 1] = buffers[offset + b];
			--final_buckets[i];
		}
	}

	free(final_buckets);
	free(buffers);
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_V2
void radix_partition_with_contiguous_buffers_streamed_256_write_v2(Tuple *input, Tuple *&output, Index *histogram, Index *unpaddedBucketSizes, const UInt buffered_tuples) {
    constexpr UInt shift = shift_distance();
	__attribute__((aligned(64))) Index *final_buckets;
	__attribute__((aligned(64))) Tuple *buffers;
	__attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];

	posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));
	posix_memalign((void**)&buffers, 64, N_PARTITIONS * buffered_tuples * sizeof(Tuple));
	memset(buffer_counters, 0, N_PARTITIONS * sizeof(UInt));

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	memcpy(unpaddedBucketSizes, final_buckets, N_PARTITIONS * sizeof(Index));

	// we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
	// if this is not the case for a partition, we add padding elements
	size_t rest;
	for(Index i = 0; i < N_PARTITIONS; ++i){
		if((rest = (final_buckets[i] % TUPLES_PER_CACHELINE)) != 0) {
			final_buckets[i] += TUPLES_PER_CACHELINE - rest;
		}
	}

	for(Index i = 1; i < N_PARTITIONS; ++i){
		final_buckets[i] += final_buckets[i - 1];
	}

	memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

	// allocate output according to histogram
	posix_memalign((void**)&output, 64, final_buckets[N_PARTITIONS - 1] * sizeof(Tuple));

	//_mm_stream_si64 (__int64* mem_addr, __int64 a)
	__attribute__((aligned(64))) Index bucket_num = 0;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		Index offset = bucket_num * buffered_tuples;
		buffers[offset + buffer_counters[bucket_num]++] = input[j];
		if (buffer_counters[bucket_num] == buffered_tuples) {
			final_buckets[bucket_num] -= buffered_tuples;
			for(UInt b = 0; b < buffered_tuples; b += STREAM_UNIT) {
				_mm256_stream_si256(reinterpret_cast<__m256i*>(output + final_buckets[bucket_num]), _mm256_load_si256((reinterpret_cast<__m256i*>(buffers + offset + b))));
				final_buckets[bucket_num] += STREAM_UNIT;
			}
			final_buckets[bucket_num] -= buffered_tuples;
			buffer_counters[bucket_num] = 0;
		}
	}

	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index offset = i * buffered_tuples;
		for (UInt b = 0; b < buffer_counters[i]; b++) {
			output[final_buckets[i] - 1] = buffers[offset + b];
			--final_buckets[i];
		}
	}

	free(final_buckets);
	free(buffers);
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_V2_UNPADDED
void radix_partition_with_contiguous_buffers_streamed_256_write_v2_unpadded(Tuple *input, Tuple *output, Index *histogram, const UInt buffered_tuples) {
    constexpr UInt shift = shift_distance();
    __attribute__((aligned(64))) Index *final_buckets;
    __attribute__((aligned(64))) Tuple *buffers;
    __attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];

    posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
    memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));
    posix_memalign((void**)&buffers, 64, N_PARTITIONS * buffered_tuples * sizeof(Tuple));
    memset(buffer_counters, 0, N_PARTITIONS * sizeof(UInt));

    for(Index j = 0; j < ELEMS; ++j){
        ++final_buckets[GET_BUCKET(input[j].value, shift)];
    }
    for(Index i = 1; i < N_PARTITIONS; ++i){
        final_buckets[i] += final_buckets[i - 1];
    }

    memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

    // we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
    // if this is not the case for a partition, we add padding elements
    for(Index i = 0; i < N_PARTITIONS; ++i){
        final_buckets[i] = final_buckets[i] - final_buckets[i]%buffered_tuples;
    }

    __attribute__((aligned(64))) Index bucket_num = 0;
    for(Index j = 0; j < ELEMS; ++j){
        bucket_num = GET_BUCKET(input[j].value, shift);
        Index offset = bucket_num * buffered_tuples;
        buffers[offset + buffer_counters[bucket_num]++] = input[j];
        if (buffer_counters[bucket_num] == buffered_tuples) {
            final_buckets[bucket_num] -= buffered_tuples;
            for(UInt b = 0; b < buffered_tuples; b += STREAM_UNIT) {
                _mm256_stream_si256(reinterpret_cast<__m256i*>(output + final_buckets[bucket_num]), _mm256_load_si256((reinterpret_cast<__m256i*>(buffers + offset + b))));
                final_buckets[bucket_num] += STREAM_UNIT;
            }
            final_buckets[bucket_num] -= buffered_tuples;
            buffer_counters[bucket_num] = 0;
        }
    }

    for (int i = N_PARTITIONS - 1; i >= 0; i--) {
        Index offset = i * buffered_tuples;
        if (i > 0 && final_buckets[i] < histogram[i - 1]) {
            //fix the wrongly written elements
            UInt end_partition = histogram[i] - 1;
            for(UInt j = final_buckets[i]; j < histogram[i - 1]; j++) {
                output[end_partition--] = output[j];
            }
            final_buckets[i] = end_partition + 1;
        }
        for (UInt b = 0; b < buffer_counters[i]; b++) {
            //rollback to end after completing the unpadded writes in the beginning
            if( final_buckets[i] <= (i > 0 ? histogram[i - 1] : 0) ) {
                final_buckets[i] = histogram[i];
            }
            output[final_buckets[i] - 1] = buffers[offset + b];
            --final_buckets[i];
        }
    }

    free(final_buckets);
    free(buffers);
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_V2_UNPADDED_PREFETCHED
void radix_partition_with_contiguous_buffers_streamed_256_write_v2_unpadded_prefetched(Tuple *input, Tuple *output, Index *histogram, const UInt buffered_tuples) {
    constexpr UInt shift = shift_distance();
    __attribute__((aligned(64))) Index *final_buckets;
    __attribute__((aligned(64))) Tuple *buffers;
    __attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];

    posix_memalign((void**)&final_buckets, 64, N_PARTITIONS * sizeof(Index));
    memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));
    posix_memalign((void**)&buffers, 64, N_PARTITIONS * buffered_tuples * sizeof(Tuple));
    memset(buffer_counters, 0, N_PARTITIONS * sizeof(UInt));

    for(Index j = 0; j < ELEMS; ++j){
        ++final_buckets[GET_BUCKET(input[j].value, shift)];
    }
    for(Index i = 1; i < N_PARTITIONS; ++i){
        final_buckets[i] += final_buckets[i - 1];
    }

    memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

    // we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
    // if this is not the case for a partition, we add padding elements
    for(Index i = 0; i < N_PARTITIONS; ++i){
        final_buckets[i] = final_buckets[i] - final_buckets[i]%buffered_tuples;
    }

    __attribute__((aligned(64))) Index bucket_num = 0;
    for(Index j = 0; j < ELEMS; ++j){
        bucket_num = GET_BUCKET(input[j].value, shift);
        Index offset = bucket_num * buffered_tuples;
        buffers[offset + buffer_counters[bucket_num]++] = input[j];
        __builtin_prefetch(buffers + offset + buffer_counters[bucket_num], 1, 0);
        if (buffer_counters[bucket_num] == buffered_tuples) {
            for(UInt b = 0; b < buffered_tuples; b += 1) {
                __builtin_prefetch(buffers + offset + b, 1, 0);
            }
            final_buckets[bucket_num] -= buffered_tuples;
            for(UInt b = 0; b < buffered_tuples; b += STREAM_UNIT) {
                _mm256_stream_si256(reinterpret_cast<__m256i*>(output + final_buckets[bucket_num]), _mm256_load_si256((reinterpret_cast<__m256i*>(buffers + offset + b))));
                final_buckets[bucket_num] += STREAM_UNIT;
            }
            final_buckets[bucket_num] -= buffered_tuples;
            buffer_counters[bucket_num] = 0;
        }
    }

    for (int i = N_PARTITIONS - 1; i >= 0; i--) {
        Index offset = i * buffered_tuples;
        if (i > 0 && final_buckets[i] < histogram[i - 1]) {
            //fix the wrongly written elements
            UInt end_partition = histogram[i] - 1;
            for(UInt j = final_buckets[i]; j < histogram[i - 1]; j++) {
                output[end_partition--] = output[j];
            }
            final_buckets[i] = end_partition + 1;
        }
        for (UInt b = 0; b < buffer_counters[i]; b++) {
            //rollback to end after completing the unpadded writes in the beginning
            if( final_buckets[i] <= (i > 0 ? histogram[i - 1] : 0) ) {
                final_buckets[i] = histogram[i];
            }
            output[final_buckets[i] - 1] = buffers[offset + b];
            --final_buckets[i];
        }
    }

    free(final_buckets);
    free(buffers);
}
#endif

static inline void
store_nontemp_64B(void * dst, void * src)
{
    register __m256i * d1 = (__m256i*) dst;
    register __m256i s1 = *((__m256i*) src);
    register __m256i * d2 = d1+1;
    register __m256i s2 = *(((__m256i*) src)+1);

    _mm256_stream_si256(d1, s1);
    _mm256_stream_si256(d2, s2);
}

#define ALIGN_NUMTUPLES(N) ((N + TUPLES_PER_CACHELINE - 1) & ~(TUPLES_PER_CACHELINE - 1))

typedef struct {
    Tuple tuples[TUPLES_PER_CACHELINE - 1];
    UInt slot;
    UInt target;
} cacheline_t;

#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental(Tuple * restrict input, Tuple *& restrict output, Index * restrict histogram, Index * restrict unpaddedBucketSizes, const UInt buffered_tuples) {
    __attribute__((aligned(64))) Index final_buckets[N_PARTITIONS];
	__attribute__((aligned(64))) cacheline_t buffers[N_PARTITIONS];

	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

	for(Index i = 0; i < N_PARTITIONS; ++i) {
		buffers[i].slot = 0;
	}

	constexpr UInt shift = shift_distance();

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	memcpy(unpaddedBucketSizes, final_buckets, N_PARTITIONS * sizeof(Index));

	// we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
	// if this is not the case for a partition, we add padding elements
	// actually, we align to the size of a full cache line, to have each flush cache line aligned
	Index offset = 0;
	for(Index i = 0; i < N_PARTITIONS; ++i){
		buffers[i].target = offset;
		offset += ALIGN_NUMTUPLES(final_buckets[i]);
	}

	for(Index i = 0; i < N_PARTITIONS; ++i){
		histogram[i] = buffers[i].target;
	}

	// allocate output according to histogram (including padding)
	posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple) + 64 * N_PARTITIONS);

	Index bucket_num = 0;

	Tuple* readStream = input;
	UInt targetBackup;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		Tuple* tuple = (Tuple*) (buffers + bucket_num);
		if(buffers[bucket_num].slot == TUPLES_PER_CACHELINE - 1) {
			targetBackup = buffers[bucket_num].target;
			tuple[TUPLES_PER_CACHELINE - 1] = *readStream;
			store_nontemp_64B(output + targetBackup, buffers + bucket_num);
			targetBackup += TUPLES_PER_CACHELINE;
			// restore
			buffers[bucket_num].slot = 0;
			buffers[bucket_num].target = targetBackup;
		}
		else {
			tuple[buffers[bucket_num].slot] = *readStream;
			++buffers[bucket_num].slot;
		}
		++readStream;
	}

	for (Index i = 0; i < N_PARTITIONS; i++) {
		for (UInt b = 0; b < buffers[i].slot; b++) {
			output[buffers[i].target++] = buffers[i].tuples[b];
		}
	}
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_PREFETCHED
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_prefetched(Tuple * restrict input, Tuple *& restrict output, Index * restrict histogram, Index * restrict unpaddedBucketSizes, const UInt buffered_tuples) {
    __attribute__((aligned(64))) Index final_buckets[N_PARTITIONS];
    __attribute__((aligned(64))) cacheline_t buffers[N_PARTITIONS];

    memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

    for(Index i = 0; i < N_PARTITIONS; ++i) {
        buffers[i].slot = 0;
    }

    constexpr UInt shift = shift_distance();

    for(Index j = 0; j < ELEMS; ++j){
        ++final_buckets[GET_BUCKET(input[j].value, shift)];
    }

    memcpy(unpaddedBucketSizes, final_buckets, N_PARTITIONS * sizeof(Index));

    // we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
    // if this is not the case for a partition, we add padding elements
    // actually, we align to the size of a full cache line, to have each flush cache line aligned
    Index offset = 0;
    for(Index i = 0; i < N_PARTITIONS; ++i){
        buffers[i].target = offset;
        offset += ALIGN_NUMTUPLES(final_buckets[i]);
    }

    for(Index i = 0; i < N_PARTITIONS; ++i){
        histogram[i] = buffers[i].target;
    }

    // allocate output according to histogram (including padding)
    posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple) + 64 * N_PARTITIONS);

    Index bucket_num = 0;

    Tuple* readStream = input;
    UInt targetBackup;
    for(Index j = 0; j < ELEMS; ++j){
        bucket_num = GET_BUCKET(input[j].value, shift);
        Tuple* tuple = (Tuple*) (buffers + bucket_num);
        __builtin_prefetch(buffers + offset + buffers[bucket_num].slot + 1, 1, 0);
        if(buffers[bucket_num].slot == TUPLES_PER_CACHELINE - 1) {
//            for(UInt b = 0; b < TUPLES_PER_CACHELINE; b += 1) {
//                __builtin_prefetch(buffers + offset + b, 1, 0);
//            }
            targetBackup = buffers[bucket_num].target;
            tuple[TUPLES_PER_CACHELINE - 1] = *readStream;
            store_nontemp_64B(output + targetBackup, buffers + bucket_num);
            targetBackup += TUPLES_PER_CACHELINE;
            // restore
            buffers[bucket_num].slot = 0;
            buffers[bucket_num].target = targetBackup;
        }
        else {
            tuple[buffers[bucket_num].slot] = *readStream;
            ++buffers[bucket_num].slot;
        }
        ++readStream;
    }

    for (Index i = 0; i < N_PARTITIONS; i++) {
        for (UInt b = 0; b < buffers[i].slot; b++) {
            output[buffers[i].target++] = buffers[i].tuples[b];
        }
    }
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_UNPADDED
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_unpadded(Tuple * restrict input, Tuple * restrict output, Index * restrict histogram, const UInt buffered_tuples) {
    __attribute__((aligned(64))) Index final_buckets[N_PARTITIONS];
    __attribute__((aligned(64))) cacheline_t buffers[N_PARTITIONS];
    constexpr UInt shift = shift_distance();

    memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

    for(Index j = 0; j < ELEMS; ++j){
        ++final_buckets[GET_BUCKET(input[j].value, shift)];
    }

    for(Index i = 1; i < N_PARTITIONS; ++i){
        final_buckets[i] += final_buckets[i - 1];
    }

    memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

    for(Index i = 0; i < N_PARTITIONS; ++i) {
        buffers[i].slot = 0;
    }

    for(Index i = 0; i < N_PARTITIONS; ++i){
        buffers[i].target = final_buckets[i] - final_buckets[i]%buffered_tuples;
    }

    Index bucket_num = 0;
    Tuple* readStream = input;
    UInt targetBackup;
    for(Index j = 0; j < ELEMS; ++j){
        bucket_num = GET_BUCKET(input[j].value, shift);
        Tuple* tuple = (Tuple*) (buffers + bucket_num);
        if(buffers[bucket_num].slot == TUPLES_PER_CACHELINE - 1) {
            targetBackup = buffers[bucket_num].target - TUPLES_PER_CACHELINE;
            tuple[TUPLES_PER_CACHELINE - 1] = *readStream;
            store_nontemp_64B(output + targetBackup, buffers + bucket_num);
            // restore
            buffers[bucket_num].slot = 0;
            buffers[bucket_num].target = targetBackup;
        }
        else {
            tuple[buffers[bucket_num].slot] = *readStream;
            ++buffers[bucket_num].slot;
        }
        ++readStream;
    }

    for (int i = N_PARTITIONS - 1; i >= 0; i--) {
//        Index offset = i * buffered_tuples;
        if (i > 0 && buffers[i].target < histogram[i - 1]) {
            //fix the wrongly written elements
            UInt end_partition = histogram[i] - 1;
            for(UInt j = buffers[i].target; j < histogram[i - 1]; j++) {
                output[end_partition--] = output[j];
            }
            buffers[i].target = end_partition + 1;
        }
        for (UInt b = 0; b < buffers[i].slot; b++) {
            //rollback to end after completing the unpadded writes in the beginning
            if( buffers[i].target <= (i > 0 ? histogram[i - 1] : 0) ) {
                buffers[i].target = histogram[i];
            }
            output[buffers[i].target - 1] = buffers[i].tuples[b];
            --buffers[i].target;
        }
    }
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_FILLSTATE
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_fillstate(Tuple * restrict input, Tuple *& restrict output, Index * restrict histogram, Index * restrict unpaddedBucketSizes, const UInt buffered_tuples) {
	__attribute__((aligned(64))) Index final_buckets[N_PARTITIONS];
	__attribute__((aligned(64))) cacheline_t buffers[N_PARTITIONS];
	__attribute__((aligned(64))) UInt targets[N_PARTITIONS];

	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

	for(Index i = 0; i < N_PARTITIONS; ++i) {
		buffers[i].slot = 0;
	}

	constexpr UInt shift = 32 - log2partitions();

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	memcpy(unpaddedBucketSizes, final_buckets, N_PARTITIONS * sizeof(Index));

	// we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
	// if this is not the case for a partition, we add padding elements
	// actually, we align to the size of a full cache line, to have each flush cache line aligned
	Index offset = 0;
	for(Index i = 0; i < N_PARTITIONS; ++i){
		targets[i] = offset;
		offset += ALIGN_NUMTUPLES(final_buckets[i]);
	}

	for(Index i = 0; i < N_PARTITIONS; ++i){
		histogram[i] = targets[i];
	}

	// allocate output according to histogram (including padding)
	posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple) + 64 * N_PARTITIONS);

	Index bucket_num = 0;

	Tuple* readStream = input;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		Tuple* tuple = (Tuple*) (buffers + bucket_num);
		if(buffers[bucket_num].slot == 8 - 1) {
			tuple[8 - 1] = *readStream;
			store_nontemp_64B(output + targets[bucket_num], buffers + bucket_num);
			targets[bucket_num] += 8;
			// restore
			buffers[bucket_num].slot = 0;
		}
		else {
			tuple[buffers[bucket_num].slot] = *readStream;
			++buffers[bucket_num].slot;
		}
		++readStream;
	}

	for (Index i = 0; i < N_PARTITIONS; i++) {
		for (UInt b = 0; b < buffers[i].slot; b++) {
			output[targets[i]++] = buffers[i].tuples[b];
		}
	}
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_FILLSTATE_UNPADDED
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_fillstate_unpadded(Tuple * restrict input, Tuple * restrict output, Index * restrict histogram, const UInt buffered_tuples) {
    __attribute__((aligned(64))) Index final_buckets[N_PARTITIONS];
    __attribute__((aligned(64))) cacheline_t buffers[N_PARTITIONS];
    __attribute__((aligned(64))) UInt targets[N_PARTITIONS];
    constexpr UInt shift = shift_distance();

    memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

    for(Index j = 0; j < ELEMS; ++j){
        ++final_buckets[GET_BUCKET(input[j].value, shift)];
    }

    for(Index i = 1; i < N_PARTITIONS; ++i){
        final_buckets[i] += final_buckets[i - 1];
    }

    memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

    for(Index i = 0; i < N_PARTITIONS; ++i) {
        buffers[i].slot = 0;
    }

    for(Index i = 0; i < N_PARTITIONS; ++i){
        targets[i] = final_buckets[i] - final_buckets[i]%buffered_tuples;
    }

    Index bucket_num = 0;

    Tuple* readStream = input;
    for(Index j = 0; j < ELEMS; ++j){
        bucket_num = GET_BUCKET(input[j].value, shift);
        Tuple* tuple = (Tuple*) (buffers + bucket_num);
        if(buffers[bucket_num].slot == TUPLES_PER_CACHELINE - 1) {
            tuple[TUPLES_PER_CACHELINE - 1] = *readStream;
            targets[bucket_num] -= TUPLES_PER_CACHELINE;
            store_nontemp_64B(output + targets[bucket_num], buffers + bucket_num);
            // restore
            buffers[bucket_num].slot = 0;
        }
        else {
            tuple[buffers[bucket_num].slot] = *readStream;
            ++buffers[bucket_num].slot;
        }
        ++readStream;
    }

    for (int i = N_PARTITIONS - 1; i >= 0; i--) {
        if (i > 0 && targets[i] < histogram[i - 1]) {
            //fix the wrongly written elements
            UInt end_partition = histogram[i] - 1;
            for(UInt j = targets[i]; j < histogram[i - 1]; j++) {
                output[end_partition--] = output[j];
            }
            targets[i] = end_partition + 1;
        }
        for (UInt b = 0; b < buffers[i].slot; b++) {
            //rollback to end after completing the unpadded writes in the beginning
            if( targets[i] <= (i > 0 ? histogram[i - 1] : 0) ) {
                targets[i] = histogram[i];
            }
            output[targets[i] - 1] = buffers[i].tuples[b];
            --targets[i];
        }
    }

}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_NOTHING
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_nothing(Tuple * restrict input, Tuple *& restrict output, Index * restrict histogram, Index * restrict unpaddedBucketSizes, const UInt buffered_tuples) {
	__attribute__((aligned(64))) Index final_buckets[N_PARTITIONS];
	__attribute__((aligned(64))) cacheline_t buffers[N_PARTITIONS];
	__attribute__((aligned(64))) UInt targets[N_PARTITIONS];
	__attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];

	memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

	for(Index i = 0; i < N_PARTITIONS; ++i) {
		buffer_counters[i] = 0;
	}

	constexpr UInt shift = 32 - log2partitions();

	for(Index j = 0; j < ELEMS; ++j){
		++final_buckets[GET_BUCKET(input[j].value, shift)];
	}

	memcpy(unpaddedBucketSizes, final_buckets, N_PARTITIONS * sizeof(Index));

	// we have to make sure that the size of each partitioning in terms of elements is a multiple of 4 (since 4 tuples fit into 32 bytes = 256 bits).
	// if this is not the case for a partition, we add padding elements
	// actually, we align to the size of a full cache line, to have each flush cache line aligned
	Index offset = 0;
	for(Index i = 0; i < N_PARTITIONS; ++i){
		targets[i] = offset;
		offset += ALIGN_NUMTUPLES(final_buckets[i]);
	}

	for(Index i = 0; i < N_PARTITIONS; ++i){
		histogram[i] = targets[i];
	}

	// allocate output according to histogram (including padding)
	posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple) + 64 * N_PARTITIONS);

	Index bucket_num = 0;

	Tuple* readStream = input;
	for(Index j = 0; j < ELEMS; ++j){
		bucket_num = GET_BUCKET(input[j].value, shift);
		Tuple* tuple = (Tuple*) (buffers + bucket_num);
		if(buffer_counters[bucket_num] == 8 - 1) {
			tuple[8 - 1] = *readStream;
			store_nontemp_64B(output + targets[bucket_num], buffers + bucket_num);
			targets[bucket_num] += 8;
			buffer_counters[bucket_num] = 0;
		}
		else {
			tuple[buffer_counters[bucket_num]] = *readStream;
			++buffer_counters[bucket_num];
		}
		++readStream;
	}

	for (Index i = 0; i < N_PARTITIONS; i++) {
		for (UInt b = 0; b < buffer_counters[i]; b++) {
			output[targets[i]++] = buffers[i].tuples[b];
		}
	}
}
#endif

#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_NOTHING_UNPADDED
void radix_partition_with_contiguous_buffers_streamed_256_write_experimental_nothing_unpadded(Tuple * restrict input, Tuple * restrict output, Index * restrict histogram, const UInt buffered_tuples) {
    __attribute__((aligned(64))) Index final_buckets[N_PARTITIONS];
    __attribute__((aligned(64))) cacheline_t buffers[N_PARTITIONS];
    __attribute__((aligned(64))) UInt targets[N_PARTITIONS];
    __attribute__((aligned(64))) UInt buffer_counters[N_PARTITIONS];
    constexpr UInt shift = shift_distance();

    memset(final_buckets, 0, N_PARTITIONS * sizeof(Index));

    for(Index j = 0; j < ELEMS; ++j){
        ++final_buckets[GET_BUCKET(input[j].value, shift)];
    }

    for(Index i = 1; i < N_PARTITIONS; ++i){
        final_buckets[i] += final_buckets[i - 1];
    }

    memcpy(histogram, final_buckets, N_PARTITIONS * sizeof(Index));

    for(Index i = 0; i < N_PARTITIONS; ++i) {
        buffer_counters[i] = 0;
    }

    for(Index i = 0; i < N_PARTITIONS; ++i){
        targets[i] = final_buckets[i] - final_buckets[i]%buffered_tuples;
    }

    Index bucket_num = 0;

    Tuple* readStream = input;
    for(Index j = 0; j < ELEMS; ++j){
        bucket_num = GET_BUCKET(input[j].value, shift);
        Tuple* tuple = (Tuple*) (buffers + bucket_num);
        if(buffer_counters[bucket_num] == TUPLES_PER_CACHELINE - 1) {
            tuple[TUPLES_PER_CACHELINE - 1] = *readStream;
            targets[bucket_num] -= TUPLES_PER_CACHELINE;
            store_nontemp_64B(output + targets[bucket_num], buffers + bucket_num);
            // restore
            buffer_counters[bucket_num] = 0;
        }
        else {
            tuple[buffer_counters[bucket_num]] = *readStream;
            ++buffer_counters[bucket_num];
        }
        ++readStream;
    }

    for (int i = N_PARTITIONS - 1; i >= 0; i--) {
        if (i > 0 && targets[i] < histogram[i - 1]) {
            //fix the wrongly written elements
            UInt end_partition = histogram[i] - 1;
            for(UInt j = targets[i]; j < histogram[i - 1]; j++) {
                output[end_partition--] = output[j];
            }
            targets[i] = end_partition + 1;
        }
        for (UInt b = 0; b < buffer_counters[i]; b++) {
            //rollback to end after completing the unpadded writes in the beginning
            if( targets[i] <= (i > 0 ? histogram[i - 1] : 0) ) {
                targets[i] = histogram[i];
            }
            output[targets[i] - 1] = buffers[i].tuples[b];
            --targets[i];
        }
    }
}
#endif