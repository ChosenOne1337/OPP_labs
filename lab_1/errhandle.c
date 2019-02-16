#include "errhandle.h"
#include <stdlib.h>

#define DEFAULT_CAPACITY 32
#define CAPACITY_MULTIPLIER 2

static void **resources = NULL;
static void (**handlers)(void*) = NULL;
static ulong resNumber = 0;
static ulong resCapacity = 0;

static int increase_capacity() {
    ulong newCapacity = 0;
    if (resCapacity == 0) {
        newCapacity = DEFAULT_CAPACITY;
    } else {
        newCapacity = resCapacity * CAPACITY_MULTIPLIER;
    }

    void **newResources = realloc(resources, newCapacity * sizeof(void*));
    if (newResources == NULL) {
        return -1;
    }
    resources = newResources;

    void (**newHandlers)(void*) = realloc(handlers, newCapacity * sizeof(void (*)(void*)));
    if (newHandlers == NULL) {
        return -1;
    }
    handlers = newHandlers;

    resCapacity = newCapacity;
    return 0;
}

void free_resources(void) {
    for (ulong resIndex = 0; resIndex < resNumber; ++resIndex) {
        handlers[resIndex](resources[resIndex]);
    }
    free(resources);
    free(handlers);
    resNumber = resCapacity = 0;
}

int add_resource(void *resource, void (*handler)(void*)) {
    if (resNumber == resCapacity) {
        int errCode = increase_capacity();
        if (errCode != 0) {
            return errCode;
        }
    }

    resources[resNumber] = resource;
    handlers[resNumber] = handler;


    ++resNumber;

    return 0;
}
