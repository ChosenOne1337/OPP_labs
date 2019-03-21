#ifndef UTILS_H
#define UTILS_H

#define SUCCESS_CODE (0)
#define FAILURE_CODE (1)

int parse_long(long *val, char *line);

int get_chunk_size(int chunkIndex, int chunksNumber, int sequenceSize);
int get_chunk_offset(int chunkIndex, int chunksNumber, int sequenceSize);

#endif // UTILS_H
