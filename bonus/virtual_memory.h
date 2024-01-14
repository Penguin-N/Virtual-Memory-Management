#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

// A entry in Invert page table
typedef struct PTENTRY {
  int v;
  int pid;
  int page_number;
  int recent_used_time;
}page_entry;

// A entry in Swap page tabel
typedef struct SPTENTRY {
  int pid;
  int page_number;
}s_page_entry;

struct VirtualMemory {
  uchar *buffer;
  uchar *storage;
  page_entry *invert_page_table;
  s_page_entry *swap_page_table;
  int *pagefault_num_ptr;
  int current_time;

  int PAGESIZE;
  int INVERT_PAGE_TABLE_SIZE;
  int PHYSICAL_MEM_SIZE;
  int STORAGE_SIZE;
  int PAGE_ENTRIES;
  int SWAP_PAGE_ENTRIES;

  // determing the next_spare position
  int *next_spare;
};


// TODO
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        page_entry *invert_page_table, s_page_entry *swap_page_table, int *pagefault_num_ptr, int *next_spare,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES, int SWAP_PAGE_ENTRIES);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size);

#endif
