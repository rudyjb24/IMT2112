__kernel void add_numbers(__global float* data, __global float* group_result,
 __global int* array_size,__global int* num_groups) {

   float sum = 0;
   uint global_addr, local_addr;
   int chunck,start,end;

   global_addr = get_global_id(0);
   local_addr = get_local_id(0);

   chunck = array_size[0]/num_groups[0];

   start = global_addr*chunck;

   if (global_addr < num_groups-1)
   {
       end = start + chunck;
   }
   else
   {
       end = array_size[0];
   }


    for(int i=start;i<end;i++)
    {
        sum += data[i];
    }
    group_result[global_addr] = sum;
   
}
