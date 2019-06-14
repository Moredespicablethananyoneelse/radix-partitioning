//////////////////////////////////////////////////////////////////
//                                                              //
//         On the Surprising Difficulty of Simple Things        // 
// Felix Martin Schuhknecht, Pankaj Khanchandani, Jens Dittrich //
//      Proceedings of the VLDB Endowment, Vol. 8, No. 9        // 
//                                                              //
//                         Used Code Base                       // 
//                    Version November 4, 2015                  //
//                    Information Systems Group                 //
//                       Saarland University                    //
//////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////
//                          Compilation                         //
//////////////////////////////////////////////////////////////////

g++ -O3 -std=c++11 -march=native Main.cpp RadixPartition.cpp -o RadixPartitioning

//////////////////////////////////////////////////////////////////
//                        Experimental Setup                    //
//////////////////////////////////////////////////////////////////

Simply chose the desired methods in Macros.h by uncommenting the defines. 
Further, you can set number of elements, number of partitions, and number of repetitions.











