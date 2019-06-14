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
 * Main.cpp
 *
 * Information Systems Group
 * Saarland University
 * 2014 - 2015
 */

#include <limits>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <assert.h>
#include "Macros.h"
#include "Types.h"
#include "RadixPartition.h"
#include <sys/mman.h>


using namespace std;

double time_difference(struct timeval& first, struct timeval& second) {
	double total_time = (second.tv_sec-first.tv_sec)*1000000+(second.tv_usec-first.tv_usec);
	return total_time/1000000;
}

void check_partioned_output(Tuple *output, Index *histogram);
void check_partioned_output_forward(Tuple *output, Index *histogram);
void check_partioned_output_padded(Tuple *output, Index *histogram, Index *unpaddedBucketSizes);
void check_partioned_output_padded_end(Tuple *output, Index *histogram, Index *unpaddedBucketSizes);
void check_partioned_output_padded_forward(Tuple *output, Index *histogram, Index *unpaddedBucketSizes);
void* malloc_huge(size_t size);

unsigned Log2(unsigned n) {
	int l = 0;
	while(n) {
		l++;
		n = n >> 1;
	}
	return l - 1;
}

int main(int argc, char *argv[]){

	mt19937 gen(19508);
	uniform_int_distribution<UInt> dis_key(0, numeric_limits<UInt>::max());
	uniform_int_distribution<UInt> dis_elem(0, numeric_limits<UInt>::max());

	Tuple *input;
	posix_memalign((void**)&input, 64, ELEMS * sizeof(Tuple));

	cout << "Performing " << TOTAL_RUNS << " runs" << endl << endl;

	for (int i = 0; i < TOTAL_RUNS; i++) {
		cout << "============================ Performing run " << i + 1 << " with " << ELEMS << " elements. ============================" << endl;
		for (size_t i = 0; i < ELEMS; ++i) {

			UInt elem = dis_elem(gen);

			input[i].rowId = elem;
			input[i].value = elem;
		}
		struct timeval start_time, end_time;
		Tuple *output;

		#ifdef UNBUFFERED_4KB_PAGE
		{
			output = (Tuple*)malloc_huge(ELEMS * sizeof(Tuple));
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with 4KB normal pages" << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_without_buffers(input, output, histogram);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output(output, histogram);
			cout << endl;
			munmap(output, ELEMS * sizeof(Tuple));
		}
		#endif

		#ifdef UNBUFFERED_2MB_PAGE
		{
			output = (Tuple*)malloc_huge(ELEMS * sizeof(Tuple));
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with 2MB huge pages" << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_without_buffers(input, output, histogram);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output(output, histogram);
			cout << endl;
			munmap(output, ELEMS * sizeof(Tuple));
		}
		#endif

		#ifdef UNBUFFERED
		{
			posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition" << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_without_buffers(input, output, histogram);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output(output, histogram);
			cout << endl;
			free(output);
		}
		#endif

		#ifdef UNBUFFERED_V2
		{
			posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition (V2) and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
			gettimeofday(&start_time, NULL);
			radix_partition_without_buffers_v2(input, output, histogram);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output_forward(output, histogram);
			cout << endl;
			free(output);
		}
		#endif

		#ifdef UNBUFFERED_PREFETCHED
		{
			posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition Unbuffered, Prefetched and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
			gettimeofday(&start_time, NULL);
			radix_partition_without_buffers_prefetched(input, output, histogram);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output(output, histogram);
			cout << endl;
			free(output);
		}
		#endif

		#ifdef RW_BUFFERS_256STREAM
		{
			if (argc < 2) {
				cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES>" << endl;
				return -1;
			}
			const UInt buffered_tuples = (UInt)atoi(argv[1]);
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			__attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (rw buffers, 256 AVX streamed)." << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_rw_buffers_streamed_256_write(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output_padded(output, histogram, unpaddedBucketSizes);
			cout << endl;
			free(output);
		}
		#endif

		#ifdef CONTIGUOUS_BUFFERS
		{
			if (argc < 2) {
				cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES>" << endl;
				return -1;
			}
			const UInt buffered_tuples = (UInt)atoi(argv[1]);
			posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous) and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers(input, output, histogram, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output(output, histogram);
			cout << endl;
			free(output);
		}
		#endif

        #ifdef CONTIGUOUS_BUFFERS_PREFETCHED
        {
            if (argc < 2) {
                cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES>" << endl;
                return -1;
            }
            const UInt buffered_tuples = (UInt)atoi(argv[1]);
            posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
            __attribute__((aligned(64))) Index histogram[N_PARTITIONS];
            cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, perfetched) and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
            gettimeofday(&start_time, NULL);
            radix_partition_with_contiguous_buffers_prefetched(input, output, histogram, buffered_tuples);
            gettimeofday(&end_time, NULL);
            cout << time_difference(start_time, end_time) << endl;
            check_partioned_output(output, histogram);
            cout << endl;
            free(output);
        }
        #endif

		#ifdef CONTIGUOUS_BUFFERS_STREAM
		{
			if (argc < 2) {
				cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES>" << endl;
				return -1;
			}
			const UInt buffered_tuples = (UInt)atoi(argv[1]);
			posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, streamed)." << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers_streamed_write(input, output, histogram, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output(output, histogram);
			cout << endl;
			free(output);
		}
		#endif

		#ifdef CONTIGUOUS_BUFFERS_256STREAM
		{
			if (argc < 2) {
				cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES>" << endl;
				return -1;
			}
			const UInt buffered_tuples = (UInt)atoi(argv[1]);
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			__attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed)." << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers_streamed_256_write(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output_padded(output, histogram, unpaddedBucketSizes);
			cout << endl;
			free(output);
		}
		#endif

		#ifdef CONTIGUOUS_BUFFERS_256STREAM_V2
		{
			if (argc < 2) {
				cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES>" << endl;
				return -1;
			}
			const UInt buffered_tuples = (UInt)atoi(argv[1]);
//			posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			__attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed) and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers_streamed_256_write_v2(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			//TODO: change check
			check_partioned_output_padded(output, histogram, unpaddedBucketSizes);
			cout << endl;
			free(output);
		}
		#endif

        #ifdef CONTIGUOUS_BUFFERS_256STREAM_V2_UNPADDED
        {
            if (argc < 2) {
                cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES in multiples of " << TUPLES_PER_CACHELINE << ">" <<endl;
                return -1;
            }
            const UInt buffered_tuples = (UInt)atoi(argv[1]);
            if (buffered_tuples % TUPLES_PER_CACHELINE > 0) {
                cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES in multiples of " << TUPLES_PER_CACHELINE << ">" <<endl;
                return -1;
            }
            posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
            __attribute__((aligned(64))) Index histogram[N_PARTITIONS];
            cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed, unpadded) and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
            gettimeofday(&start_time, NULL);
            radix_partition_with_contiguous_buffers_streamed_256_write_v2_unpadded(input, output, histogram, buffered_tuples);
            gettimeofday(&end_time, NULL);
            cout << time_difference(start_time, end_time) << endl;
            check_partioned_output(output, histogram);
            cout << endl;
            free(output);
        }
        #endif

        #ifdef CONTIGUOUS_BUFFERS_256STREAM_V2_UNPADDED_PREFETCHED
        {
            if (argc < 2) {
                cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES in multiples of " << TUPLES_PER_CACHELINE << ">" <<endl;
                return -1;
            }
            const UInt buffered_tuples = (UInt)atoi(argv[1]);
            if (buffered_tuples % TUPLES_PER_CACHELINE > 0) {
                cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES in multiples of " << TUPLES_PER_CACHELINE << ">" <<endl;
                return -1;
            }
            posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
            __attribute__((aligned(64))) Index histogram[N_PARTITIONS];
            cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed, unpadded, prefetched) and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
            gettimeofday(&start_time, NULL);
            radix_partition_with_contiguous_buffers_streamed_256_write_v2_unpadded_prefetched(input, output, histogram, buffered_tuples);
            gettimeofday(&end_time, NULL);
            cout << time_difference(start_time, end_time) << endl;
            check_partioned_output(output, histogram);
            cout << endl;
            free(output);
        }
        #endif

		#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL
		{
			const UInt buffered_tuples = TUPLES_PER_CACHELINE;
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			__attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed) both and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers_streamed_256_write_experimental(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output_padded_forward(output, histogram, unpaddedBucketSizes);
			cout << endl;
			free(output);
		}
		#endif

        #ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_PREFETCHED
        {
            const UInt buffered_tuples = TUPLES_PER_CACHELINE;
            //posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
            __attribute__((aligned(64))) Index histogram[N_PARTITIONS];
            __attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
            cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed, prefetched) both and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
            gettimeofday(&start_time, NULL);
            radix_partition_with_contiguous_buffers_streamed_256_write_experimental_prefetched(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
            gettimeofday(&end_time, NULL);
            cout << time_difference(start_time, end_time) << endl;
            check_partioned_output_padded_forward(output, histogram, unpaddedBucketSizes);
            cout << endl;
            free(output);
        }
        #endif



        #ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_UNPADDED
        {
            const UInt buffered_tuples = TUPLES_PER_CACHELINE;
            posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
            __attribute__((aligned(64))) Index histogram[N_PARTITIONS];
            cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed, unpadded, micro-row) both and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
            gettimeofday(&start_time, NULL);
            radix_partition_with_contiguous_buffers_streamed_256_write_experimental_unpadded(input, output, histogram, buffered_tuples);
            gettimeofday(&end_time, NULL);
            cout << time_difference(start_time, end_time) << endl;
            check_partioned_output(output, histogram);
            cout << endl;
            free(output);
        }
        #endif

		#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_FILLSTATE
		{
			const UInt buffered_tuples = 8;
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			__attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed) just fillstate." << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers_streamed_256_write_experimental_fillstate(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			//TODO: change check
			check_partioned_output_padded_forward(output, histogram, unpaddedBucketSizes);
			cout << endl;
			free(output);
		}
		#endif

        #ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_FILLSTATE_UNPADDED
        {
            const UInt buffered_tuples = TUPLES_PER_CACHELINE;
            posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
            __attribute__((aligned(64))) Index histogram[N_PARTITIONS];
            cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed, unpadded, micro-row-fillstate-only) both and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
            gettimeofday(&start_time, NULL);
            radix_partition_with_contiguous_buffers_streamed_256_write_experimental_fillstate_unpadded(input, output, histogram, buffered_tuples);
            gettimeofday(&end_time, NULL);
            cout << time_difference(start_time, end_time) << endl;
            check_partioned_output(output, histogram);
            cout << endl;
            free(output);
        }
        #endif

		#ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_NOTHING
		{
			const UInt buffered_tuples = 8;
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			__attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed) nothing." << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers_streamed_256_write_experimental_nothing(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			//TODO: change check
			check_partioned_output_padded_forward(output, histogram, unpaddedBucketSizes);
			cout << endl;
			free(output);
		}
		#endif

        #ifdef CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_NOTHING_UNPADDED
        {
            const UInt buffered_tuples = TUPLES_PER_CACHELINE;
            posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
            __attribute__((aligned(64))) Index histogram[N_PARTITIONS];
            cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed, unpadded, micro-row-nothing) both and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
            gettimeofday(&start_time, NULL);
            radix_partition_with_contiguous_buffers_streamed_256_write_experimental_nothing_unpadded(input, output, histogram, buffered_tuples);
            gettimeofday(&end_time, NULL);
            cout << time_difference(start_time, end_time) << endl;
            check_partioned_output(output, histogram);
            cout << endl;
            free(output);
        }
        #endif

		#ifdef CONTIGUOUS_BUFFERS_256STREAM_PREFETCHED
		{
			if (argc < 2) {
				cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES>" << endl;
				return -1;
			}
			const UInt buffered_tuples = (UInt)atoi(argv[1]);
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			__attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed, prefetched)." << endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers_streamed_256_write_prefetched(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output_padded(output, histogram, unpaddedBucketSizes);
			cout << endl;
			free(output);
		}
		#endif

		#ifdef CONTIGUOUS_BUFFERS_256STREAM_PREFETCHED_V2
		{
			if (argc < 2) {
				cout << "Usage: " << argv[0] << " <BUFFERED_TUPLES>" << endl;
				return -1;
			}
			const UInt buffered_tuples = (UInt)atoi(argv[1]);
			//posix_memalign((void**)&output, 64, ELEMS * sizeof(Tuple));
			__attribute__((aligned(64))) Index histogram[N_PARTITIONS];
			__attribute__((aligned(64))) Index unpaddedBucketSizes[N_PARTITIONS];
			cout << "Starting Single Threaded Radix Partition with " << buffered_tuples << " buffered tuples per partition (contiguous, 256 AVX streamed, prefetched) and size of Tuple as " << sizeof(Tuple) << " bytes."<<endl;
			gettimeofday(&start_time, NULL);
			radix_partition_with_contiguous_buffers_streamed_256_write_prefetched_v2(input, output, histogram, unpaddedBucketSizes, buffered_tuples);
			gettimeofday(&end_time, NULL);
			cout << time_difference(start_time, end_time) << endl;
			check_partioned_output_padded(output, histogram, unpaddedBucketSizes);
			cout << endl;
			free(output);
		}
		#endif
	}

}

void check_partioned_output(Tuple *output, Index *histogram) {
#ifdef DEBUG_CHECK
	constexpr UInt shift = shift_distance();
	long long checksum = 0;
	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index start = (i == 0 ? 0 : histogram[i - 1]);
		Index end = histogram[i];
		for (Index j = start; j < end; j++) {
			if (GET_BUCKET(output[j].value, shift) != i) {
				cout << "INCORRECT ===>" << endl;
				return;
			} else {
				checksum += output[j].value * i;
			}
		}
	}
	cout << "Correct !" << endl;
	cout << "Checksum " << checksum << endl;
#endif
}

void check_partioned_output_forward(Tuple *output, Index *histogram) {
#ifdef DEBUG_CHECK
	constexpr UInt shift = shift_distance();
	long long checksum = 0;
	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index start = histogram[i];
		Index end = i == N_PARTITIONS - 1 ? ELEMS : histogram[i+1];
		for (Index j = start; j < end; j++) {
			if (GET_BUCKET(output[j].value, shift) != i) {
				cout << "INCORRECT ===>" << endl;
				return;
			} else {
				checksum += output[j].value * i;
			}
		}
	}
	cout << "Correct !" << endl;
	cout << "Checksum " << checksum << endl;
#endif
}

void check_partioned_output_padded(Tuple *output, Index *histogram, Index *unpaddedBucketSizes) {
#ifdef DEBUG_CHECK
	constexpr UInt shift = shift_distance();
	long long checksum = 0;
	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index start = histogram[i] - unpaddedBucketSizes[i];
		Index end = histogram[i];
		for (Index j = start; j < end; j++) {
			if (GET_BUCKET(output[j].value, shift) != i) {
				cout << "INCORRECT ===>" << endl;
				return;
			} else {
				checksum += output[j].value * i;
			}
		}
	}
	cout << "Correct !" << endl;
	cout << "Checksum " << checksum << endl;
#endif
}

void check_partioned_output_padded_end(Tuple *output, Index *histogram, Index *unpaddedBucketSizes) {
#ifdef DEBUG_CHECK
	constexpr UInt shift = shift_distance();
	long long checksum = 0;
	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index start = i > 0 ? histogram[i - 1] : 0;
		Index end = start + unpaddedBucketSizes[i];
		for (Index j = start; j < end; j++) {
			if (GET_BUCKET(output[j].value, shift) != i) {
				cout << "INCORRECT ===>" << endl;
				return;
			} else {
				checksum += output[j].value * i;
			}
		}
	}
	cout << "Correct !" << endl;
	cout << "Checksum " << checksum << endl;
#endif
}

void check_partioned_output_padded_forward(Tuple *output, Index *histogram, Index *unpaddedBucketSizes) {
#ifdef DEBUG_CHECK
	constexpr UInt shift = shift_distance();
	long long checksum = 0;
	for (Index i = 0; i < N_PARTITIONS; i++) {
		Index start = histogram[i];
		Index end = histogram[i] + unpaddedBucketSizes[i];
		for (Index j = start; j < end; j++) {
			if (GET_BUCKET(output[j].value, shift) != i) {
				cout << "INCORRECT ===>" << endl;
				return;
			} else {
				checksum += output[j].value * i;
			}
		}
	}
	cout << "Correct !" << endl;
	cout << "Checksum " << checksum << endl;
#endif
}

#ifndef MAP_ANONYMOUS
# define MAP_ANONYMOUS MAP_ANON
#endif

void* malloc_huge_transparent(size_t size) {
	void* p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
	if(!p) {
		std::cout << "Out of memory" << std::endl;
		exit(1);
	}
	#ifndef __MACH__
		madvise(p, size, MADV_HUGEPAGE);
		madvise(p, size, MADV_SEQUENTIAL);
	#endif
	return p;
}

#if defined(UNBUFFERED_4KB_PAGE) || defined(UNBUFFERED_2MB_PAGE)
void* malloc_huge(size_t size) {
	#ifndef MAP_HUGE_1GB
	# define MAP_HUGE_1GB (30 << 26)
	#endif
	#ifdef UNBUFFERED_4KB_PAGE
		#define FLAGS  MAP_PRIVATE|MAP_ANONYMOUS
	#elif defined(UNBUFFERED_2MB_PAGE)
		#define FLAGS  MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB
	#elif  defined(UNBUFFERED_1G_PAGE)
		#define FLAGS  MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|MAP_HUGE_1GB
	#else
		std::cout << "Please set page size." <<std::endl;
		return NULL;
	#endif
	void* p = mmap(NULL, size, PROT_READ|PROT_WRITE, FLAGS, -1, 0);
	if(!p || p == MAP_FAILED) {
		std::cout << "Out of memory" << std::endl;
		exit(1);
	}
	return p;
}
#endif
