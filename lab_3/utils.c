#include "utils.h"
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

#define TRUE (1)
#define FALSE (0)
#define NO_ERROR (0)

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

static int is_whitespace_string(char *str) {
    static const char spaceCharSet[] = " \t\n\v\f\r";
    unsigned long stringLength = strlen(str);
    size_t spaceSequenceLen = strspn(str, spaceCharSet);
    if ((size_t)stringLength != spaceSequenceLen) {
        return FALSE;
    }
    return TRUE;
}


int parse_long(long *val, char *line) {
    int base = 10;
    char *lineTail = line;
    errno = NO_ERROR;
    long number = strtol(line, &lineTail, base);
    if (errno == ERANGE && (number == LONG_MAX || number == LONG_MIN)) {
        perror("parse_long(..) error");
        return FAILURE_CODE;
    }
    if (number == 0 && errno != NO_ERROR) {
        perror("parse_long(..) error");
        return FAILURE_CODE;
    }
    if (lineTail == line) {
        return FAILURE_CODE;
    }
    int isWhitespaceStr = is_whitespace_string(lineTail);
    if (isWhitespaceStr == FALSE) {
        return FAILURE_CODE;
    }

    *val = number;

    return SUCCESS_CODE;
}
