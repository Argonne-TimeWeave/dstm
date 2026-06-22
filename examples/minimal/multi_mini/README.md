# multiple-regions

This currently does not work. Support for this does not exist within the DSTM framework

## What would be needed (most likely)

A region\_id that is assigned at dstm\_region\_create time. Which each rank can hold the region\_id of the region they belong to. 

dstm\_get\_region\_by\_id

dstm get/put/darray\_create may have to use a region parameter (a region id)

versions may be fine if no communication occurs between regions. Also in this case we could have per region barriers. 

Some sort of cross-region handling may need to be added. A put/get across ranks may require some barriers and a unique put function. Cross-region distance may also be a unqiue question

*Will have to decide if regions are meant to be independent of each other OR if they just represent another level of partitions.*
