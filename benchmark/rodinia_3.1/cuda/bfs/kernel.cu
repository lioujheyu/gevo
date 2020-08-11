/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad.
All rights reserved.

Permission to use, copy, modify and distribute this software and its documentation for
educational purpose is hereby granted without fee, provided that the above copyright
notice and this permission notice appear in all copies of this software and that you do
not sell the software.

THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR
OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_

__device__ __forceinline__ bool ld_gbl_cg (const bool *addr)
{
    short t;
// #if defined(__LP64__) || defined(_WIN64)
    asm ("ld.global.cg.u8 %0, [%1];" : "=h"(t) : "l"(addr));
// #else
//     asm ("ld.global.cg.u8 %0, [%1];" : "=h"(t) : "r"(addr));
// #endif
    return (bool)t;
}

#endif

__global__ void
Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes)
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	// bool t = ld_gbl_cg(&g_graph_mask[tid]);
	// asm("ld.global.cg.u8 %0 [%1];" : "=h"(t) : "l"(u): "memory");
	if( tid<no_of_nodes && g_graph_mask[tid])
	// if( tid<no_of_nodes && t)
	{
		g_graph_mask[tid]=false;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
			{
			int id = g_graph_edges[i];
			// int id = __ldg(&g_graph_edges[i]);
			if(!g_graph_visited[id])
				{
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
				}
			}
	}
}


