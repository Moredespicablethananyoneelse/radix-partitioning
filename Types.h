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
 * Types.h
 *
 * Information Systems Group
 * Saarland University
 * 2014 - 2015
 */

#ifndef TYPES_H_
#define TYPES_H_
#include <inttypes.h>
#include "Macros.h"
typedef uint32_t Index;

#ifdef WIDTH_64
typedef unsigned long UInt;
#define TUPLES_PER_CACHELINE 4
#define STREAM_UNIT 2
#else
typedef unsigned int UInt;
#define TUPLES_PER_CACHELINE 8
#define STREAM_UNIT 4
#endif

typedef unsigned char Byte;

struct Tuple {
	UInt value;
	UInt rowId;
};

constexpr UInt shift_distance() {
    #ifdef WIDTH_64
    return (64 - log2partitions());
    #else
    return (32 - log2partitions());
    #endif
}
#endif
