#include "utils.h"

int get_chunk_size(int chunkIndex, int chunksNumber, int sequenceSize) {
    // the remainder is uniformly distributed in chunks
    int chunkSize = sequenceSize / chunksNumber;
    int remainder = sequenceSize % chunksNumber;
    return chunkSize + (chunkIndex < remainder ? 1 : 0);
}

int get_chunk_offset(int chunkIndex, int chunksNumber, int sequenceSize) {
    int chunkSize = sequenceSize / chunksNumber;
    int remainder = sequenceSize % chunksNumber;
    return chunkIndex * chunkSize + (chunkIndex < remainder ? chunkIndex : remainder);
}
