__kernel void add_numbers(__global float4* data, 
      __local float* local_result, __global float* group_result) {

   float sum;
   float4 sum_vector;
   uint global_addr, local_addr;

   global_addr = get_global_id(0);
   local_addr = get_local_id(0);
   sum_vector = data[global_addr];

   local_result[local_addr] = sum_vector.s0 + sum_vector.s1 + 
                              sum_vector.s2 + sum_vector.s3; 

   if(local_addr == 0) {
      sum = 0;
      for(int i=0; i<get_local_size(0); i++) {
         sum += local_result[i];
      }
      group_result[global_addr] = sum;
   }
}
