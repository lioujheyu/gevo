//========================================================================================================================================================================================================200
//	findK function
//========================================================================================================================================================================================================200

__global__ void
findK(	long height,
		knode *knodesD,
		long knodes_elem,
		record *recordsD,

		long *currKnodeD,
		long *offsetD,
		int *keysD,
		record *ansD)
{

	// private thread IDs
	int thid = threadIdx.x;
	int bid = blockIdx.x;

	// // processtree levels
	// int i;
	// for(i = 0; i < height; i++){

	// 	// if value is between the two keys
	// 	if((knodesD[currKnodeD[bid]].keys[thid]) <= keysD[bid] && (knodesD[currKnodeD[bid]].keys[thid+1] > keysD[bid])){
	// 		// this conditional statement is inserted to avoid crush due to but in original code
	// 		// "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
	// 		// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
	// 		if(knodesD[offsetD[bid]].indices[thid] < knodes_elem){
	// 			offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
	// 		}
	// 	}
	// 	__syncthreads();

	// 	// set for next tree level
	// 	if(thid==0){
	// 		currKnodeD[bid] = offsetD[bid];
	// 	}
	// 	__syncthreads();

	// }

	// //At this point, we have a candidate leaf node which may contain
	// //the target record.  Check each key to hopefully find the record
	// if(knodesD[currKnodeD[bid]].keys[thid] == keysD[bid]){
	// 	ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
	// }

	ansD[bid].value = keysD[bid];

}

//========================================================================================================================================================================================================200
//	findRangeK function
//========================================================================================================================================================================================================200
__global__ void
findRangeK(	long height,

			knode *knodesD,
			long knodes_elem,

			long *currKnodeD,
			long *offsetD,
			long *lastKnodeD,
			long *offset_2D,
			int *startD,
			int *endD,
			int *RecstartD,
			int *ReclenD)
{

	// private thread IDs
	int thid = threadIdx.x;
	int bid = blockIdx.x;

	// ???
	int i;
	for(i = 0; i < height; i++){

		if((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid+1] > startD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){
				offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
			}
		}
		if((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid+1] > endD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){
				offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
			}
		}
		__syncthreads();

		// set for next tree level
		if(thid==0){
			currKnodeD[bid] = offsetD[bid];
			lastKnodeD[bid] = offset_2D[bid];
		}
		__syncthreads();
	}

	// Find the index of the starting record
	if(knodesD[currKnodeD[bid]].keys[thid] == startD[bid]){
		RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
	}
	__syncthreads();

	// Find the index of the ending record
	if(knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]){
		ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1;
	}

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
