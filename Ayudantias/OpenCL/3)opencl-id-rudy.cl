__kernel void getmyid(                
    __global int *a,                  
    __global int *b,                  
    __global int *c,                  
    __global int *d,                  
    __const unsigned int n)           
{                                     
    int id = get_global_id(0);        
	                                   
    if(id < n) 
    {                      
       a[id] = get_global_id(0);      
       b[id] = get_local_id(0);       
       c[id] = get_group_id(0);       
       d[id] = get_local_size(0);     
    }                                  
}                                     
;