import pyopencl as cl

print('OpenCL Platforms')
for count, platform in enumerate(cl.get_platforms()):
    print('')
    print('Platform ' + str(count) + ' - Name:    ' + platform.name)
    print('Platform ' + str(count) + ' - Vendor:  ' + platform.vendor)
    print('Platform ' + str(count) + ' - Version: ' + platform.version)
    print('Platform ' + str(count) + ' - Profile: ' + platform.profile)

print('OpenCL Devices')
for p, platform in enumerate(cl.get_platforms()):
    print('')
    print('Platform ' + str(p) + ' - Name:  ' + platform.name)
    for d, device in enumerate(platform.get_devices()):
        print('')
        print('Device ' + str(p) + '.' + str(d) + ' - Name:  ' + device.name)
        print('Device ' + str(p) + '.' + str(d) + ' - Type:  ' + cl.device_type.to_string(device.type))
        print('Device ' + str(p) + '.' + str(d) + ' - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
        print('Device ' + str(p) + '.' + str(d) + ' - Compute Units:  {0}'.format(device.max_compute_units))
        print('Device ' + str(p) + '.' + str(d) + ' - Local Memory:  {0:.0f} KB'.format(device.local_mem_size/1024.0))
        print('Device ' + str(p) + '.' + str(d) + ' - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size/1024.0))
        print('Device ' + str(p) + '.' + str(d) + ' - Global Memory: {0:.0f} GB'.format(device.global_mem_size/1073741824.0))
        print('Device ' + str(p) + '.' + str(d) + ' - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))