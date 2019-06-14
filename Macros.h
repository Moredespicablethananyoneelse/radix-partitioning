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
 * Macros.h
 *
 * Information Systems Group
 * Saarland University
 * 2014 - 2015
 */

#ifndef MACROS_H_
#define MACROS_H_

#define ELEMS 100000000
#define N_PARTITIONS 16384
////#define WIDTH_64
#define TOTAL_RUNS 2

#define UNBUFFERED
#define UNBUFFERED_V2
#define UNBUFFERED_PREFETCHED
#define UNBUFFERED_4KB_PAGE
#define UNBUFFERED_2MB_PAGE

#define CONTIGUOUS_BUFFERS
#define CONTIGUOUS_BUFFERS_PREFETCHED
#define CONTIGUOUS_BUFFERS_256STREAM
#define CONTIGUOUS_BUFFERS_256STREAM_V2
#define CONTIGUOUS_BUFFERS_256STREAM_V2_UNPADDED
#define CONTIGUOUS_BUFFERS_256STREAM_V2_UNPADDED_PREFETCHED

#define CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL
#define CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_PREFETCHED
#define CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_UNPADDED
#define CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_FILLSTATE
#define CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_FILLSTATE_UNPADDED
#define CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_NOTHING
#define CONTIGUOUS_BUFFERS_256STREAM_EXPERIMENTAL_NOTHING_UNPADDED

#define CONTIGUOUS_BUFFERS_256STREAM_PREFETCHED
#define CONTIGUOUS_BUFFERS_256STREAM_PREFETCHED_V2

#define DEBUG_CHECK

constexpr unsigned int log2partitions() {
	return (31 - __builtin_clz(N_PARTITIONS)) * (N_PARTITIONS ? 1 : 0);
}
#define GET_BUCKET(VALUE, BITS) (VALUE >> BITS)

#endif
