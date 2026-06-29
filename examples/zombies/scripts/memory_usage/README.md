# memory metrics used

RSS is actual memory that was used by the simulation while VSZ is how much virtual memory was requested by our program. 

VSZ is most likely extremely high for DSTM because i set the default region size to be very large. 
    That also explains why it is constant as we scale.
