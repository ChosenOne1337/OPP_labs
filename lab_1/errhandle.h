#ifndef ERRHANDLE_H
#define ERRHANDLE_H

void free_resources(void);

void add_resource(void *res, void (*handler)(void*));

#endif // ERRHANDLE_H
