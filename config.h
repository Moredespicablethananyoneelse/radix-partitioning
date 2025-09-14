#ifndef CONFIG_H_
#define CONFIG_H_

#define UNBUFFERED

// Define the number of elements and partitions
#define ELEMS 100000000
#define N_PARTITIONS 1024

// Define the radix partitioning function to be used
#define log2partitions() 10

// Define the tuple width (32 or 64 bit)
#define WIDTH_64

#endif // CONFIG_H_
