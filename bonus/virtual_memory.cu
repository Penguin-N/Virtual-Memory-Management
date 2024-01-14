#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


__device__ void init_invert_page_table(VirtualMemory *vm) {
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    page_entry pe;
    pe.v = 1;
    pe.pid = 0;
    pe.page_number = 0;
    pe.recent_used_time = 0;
    vm->invert_page_table[i] = pe;
  }
}

__device__ void init_swap_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->SWAP_PAGE_ENTRIES; i++) {
    s_page_entry spe;
    spe.pid = -1;
    spe.page_number = -1;
    vm->swap_page_table[i] = spe;
  }
}


__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        page_entry *invert_page_table, s_page_entry *swap_page_table, int *pagefault_num_ptr, int *next_spare,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES, int SWAP_PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->swap_page_table = swap_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;
  vm->SWAP_PAGE_ENTRIES = SWAP_PAGE_ENTRIES;

  vm->next_spare = next_spare;
  vm->current_time = 0;

  // before first vm_write or vm_read
  if (threadIdx.x==0){
    init_invert_page_table(vm);
    init_swap_page_table(vm);
  }
}



// Search in invert page table
__device__ int search_in_pt(VirtualMemory *vm, u32 VM_address){
  int pid = threadIdx.x;
  int page_number = (VM_address >> 5) & ((1 << 13)-1); 
  int offside = VM_address & ((1<<5)-1);
  
  (vm->current_time)++;
  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    page_entry pe = vm->invert_page_table[i];
    // Find one 
    if ((pe.v==0)&&(pe.pid==pid)&&(pe.page_number==page_number)){
      return (i<<5) + offside;
    }
    // Invalid: we add new a new page.
    else if(pe.v==1){
      (*vm->pagefault_num_ptr)++;
      vm->invert_page_table[i].v = 0;
      vm->invert_page_table[i].pid = pid;
      vm->invert_page_table[i].page_number = page_number;
      vm->invert_page_table[i].recent_used_time = vm->current_time;
      return (i<<5) + offside;
    }
  }

  // Page full and Page false.
  (*vm->pagefault_num_ptr)++;
  return -1;
}


// Search in swap table
__device__ int search_in_st(VirtualMemory *vm, u32 addr){
  int pid = threadIdx.x;
  int page_number = (addr >> 5) & ((1 << 13)-1);  

  for(int i=0;i<vm->SWAP_PAGE_ENTRIES;i++){
    s_page_entry spe = vm->swap_page_table[i];
    // Find one: Remove the swap table and prepared for swapping in.
    if((spe.pid == pid)&&(spe.page_number == page_number)){
      vm->swap_page_table[i].pid = -1;
      vm->swap_page_table[i].page_number = -1;
      (*vm->next_spare) += 1;
      return i;
    }
  }

  // Cannot find one: Build a new page.
  (*vm->next_spare) += 1;
  return *vm->next_spare-1;
}

// LRU method
__device__ int lru(VirtualMemory *vm){
  int smallest = 1<<30;
  int target_idx;
  for (int i=0;i<vm->PAGE_ENTRIES;i++){
    page_entry pe = vm->invert_page_table[i];
    if (smallest > pe.recent_used_time && pe.pid == threadIdx.x){
      smallest = pe.recent_used_time;
      target_idx = i;
    }
  }
  return target_idx;
}

// Swap
__device__ void swap(VirtualMemory *vm, int pid, int page_number, int PM_page_number, int SM_page_number){
  // Bulid a new entry
  s_page_entry spe;
  spe.pid = pid;
  spe.page_number = page_number;
  vm->swap_page_table[SM_page_number] = spe;

  // Swap the content.
  int PM_address = PM_page_number << 5;
  int SM_address = SM_page_number << 5;
  for (int i=0;i<32;i++){
    uchar content = vm->buffer[i+PM_address];
    vm->buffer[i+PM_address] = vm->storage[i+SM_address];
    vm->storage[i+SM_address] = content;
  }
}


__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  if (addr%blockDim.x!=threadIdx.x) return NULL;
  addr = addr/blockDim.x;
  uchar value;
  int pid = threadIdx.x;
  int page_number = (addr >> 5) & ((1 << 13)-1); 
  int offside = addr & ((1<<5)-1); 
  int P_address = search_in_pt(vm,addr);
  // If we get the physical address.
  if (P_address!=-1){
    value = vm->buffer[P_address];
  }
  // Page full and Page fault.
  else{
    // Search in the swap table
    int SM_page_number = search_in_st(vm,addr);
    // Find target entry
    int PM_page_number = lru(vm);
    // Swap
    page_entry old_pe = vm->invert_page_table[PM_page_number];
    swap(vm,old_pe.pid,old_pe.page_number,PM_page_number,SM_page_number);
    // Update the page entry
    vm->invert_page_table[PM_page_number].v = 0;
    vm->invert_page_table[PM_page_number].pid = pid;
    vm->invert_page_table[PM_page_number].page_number = page_number;
    vm->invert_page_table[PM_page_number].recent_used_time = vm->current_time;
    // Read
    int PM_address = (PM_page_number<<5)+offside;
    value = vm->buffer[PM_address];
  }  
  return value; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  if (addr%blockDim.x!=threadIdx.x) return;
  addr = addr/blockDim.x;
  /* Complete vm_write function to write value into data buffer */
  int pid = threadIdx.x;
  int page_number = (addr >> 5) & ((1 << 13)-1); 
  int offside = addr & ((1<<5)-1); 
  int P_address = search_in_pt(vm,addr);
  // If we get the physical address.
  if (P_address!=-1){
    vm->buffer[P_address] = value;
  }
  // Page full and Page fault.
  else{
    // Search in the swap table
    int SM_page_number = search_in_st(vm,addr);
    // Find target entry
    int PM_page_number = lru(vm);
    // Swap
    page_entry old_pe = vm->invert_page_table[PM_page_number];
    swap(vm,old_pe.pid,old_pe.page_number,PM_page_number,SM_page_number);
    // Update the page entry
    vm->invert_page_table[PM_page_number].v = 0;
    vm->invert_page_table[PM_page_number].pid = pid;
    vm->invert_page_table[PM_page_number].page_number = page_number;
    vm->invert_page_table[PM_page_number].recent_used_time = vm->current_time;
    // Write
    int PM_address = (PM_page_number<<5)+offside;
    vm->buffer[PM_address] = value;
  }

}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i=0;i<input_size;i++){
    uchar value = vm_read(vm,i+offset);
    if (value==NULL) continue;
    results[i] = value;
  }

}

