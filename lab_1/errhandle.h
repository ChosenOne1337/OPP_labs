#ifndef ERRHANDLE_H
#define ERRHANDLE_H

void free_resources(void);

int add_resource(void *resource, void (*handler)(void*));

#endif // ERRHANDLE_H
