from mpi4py import MPI
from enum import Enum
import numpy as np
import uuid

from collections.abc import Buffer
from typing import final, Any
import numpy.typing as npt

###################
# GLOBALS (for now)
###################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


@final
class Context:
    """Context class that manages MPI world size and rank information."""
    
    def __init__(self, distance_matrix: npt.NDArray[np.int32]):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.memory_regions: dict[uuid.UUID, MemoryRegion] = {}
        self.uuid_to_rank_mapping: dict[uuid.UUID, int] = {}
        
        # Store distance matrix between ranks (required parameter)
        if distance_matrix.shape != (self.world_size, self.world_size):
            raise ValueError(f"Distance matrix shape {distance_matrix.shape} does not match world size {self.world_size}")
        self.distance_matrix = distance_matrix
    
    def get_distance(self, from_rank: int, to_rank: int) -> int:
        """Get the distance between two ranks.
        
        Args:
            from_rank: Source rank
            to_rank: Destination rank
            
        Returns:
            Distance (number of hops) from from_rank to to_rank
            
        Raises:
            ValueError: If either rank is invalid
        """
        if not (0 <= from_rank < self.world_size and 0 <= to_rank < self.world_size):
            raise ValueError(f"Invalid rank: from_rank={from_rank}, to_rank={to_rank}, world_size={self.world_size}")
        
        return int(self.distance_matrix[from_rank][to_rank])
    
    def create_memory_regions(self, memsize: int, mems_per_rank: int = 1):
        """Create memory regions and store them internally indexed by UUID.
        
        Args:
            memsize: Size of each memory region in bytes
            mems_per_rank: Number of memory regions per rank
            
        Returns:
            List of UUIDs for the created memory regions
        """
        self.mems_per_rank = mems_per_rank
        
        # Step 1: Create UUIDs for the local memory regions
        local_uuids = []
        for i in range(mems_per_rank):
            memory_uuid = uuid.uuid4()
            local_uuids.append(memory_uuid)
        
        # Step 2: Share UUIDs across all ranks
        all_uuids = self._share_uuids(local_uuids)
        
        # Step 3: Create a memory region per UUID on each rank
        for memory_uuid in all_uuids:
            memory_region = MemoryRegion(memory_uuid, LogicalTimestamp(), self, memsize)
            self.memory_regions[memory_uuid] = memory_region
        
        # Set owner information for all memory regions after mapping is available
        for memory_region in self.memory_regions.values():
            memory_region._set_owner_info()
        
        return all_uuids
    
    def _share_uuids(self, local_uuids: list[uuid.UUID]) -> list[uuid.UUID]:
        """Share UUIDs across all ranks using MPI alltoall.
        
        Args:
            local_uuids: List of UUIDs created by this rank
            
        Returns:
            List of all UUIDs from all ranks
        """
        # Convert UUIDs to bytes for MPI communication
        uuid_bytes_list = [uuid_obj.bytes for uuid_obj in local_uuids]
        
        # Pad all lists to the same length for alltoall
        max_uuids = self.mems_per_rank
        padded_uuid_bytes = uuid_bytes_list + [b'\x00' * 16] * (max_uuids - len(uuid_bytes_list))
        
        # Flatten the list for MPI alltoall
        send_data = b''.join(padded_uuid_bytes)
        
        # Alltoall to share UUIDs across all ranks
        recv_data = self.comm.alltoall([send_data] * self.world_size)
        
        # Build the global UUID-to-rank mapping and collect all UUIDs
        all_uuids = []
        for rank_idx, rank_data in enumerate(recv_data):
            # Parse UUIDs from the received data
            for i in range(max_uuids):
                start_idx = i * 16
                end_idx = start_idx + 16
                uuid_bytes = rank_data[start_idx:end_idx]
                
                # Skip padding (all zeros)
                if uuid_bytes != b'\x00' * 16:
                    memory_uuid = uuid.UUID(bytes=uuid_bytes)
                    self.uuid_to_rank_mapping[memory_uuid] = rank_idx
                    all_uuids.append(memory_uuid)
        
        return all_uuids

    def get_memory_region(self, memory_uuid: uuid.UUID) -> MemoryRegion:
        """Get a memory region by its UUID.
        
        Args:
            memory_uuid: The UUID of the memory region to retrieve
            
        Returns:
            The MemoryRegion object with the specified UUID
            
        Raises:
            KeyError: If no memory region exists with the given UUID
        """
        return self.memory_regions[memory_uuid]

    def get_uuids_for_rank(self, rank: int) -> list[uuid.UUID]:
        """Get all UUIDs of memory regions owned by a specific rank.
        
        Args:
            rank: The rank whose memory region UUIDs to retrieve
            
        Returns:
            List of UUIDs for memory regions owned by the specified rank
        """
        if not hasattr(self, 'mems_per_rank') or self.mems_per_rank == 0:
            return []
        
        uuids = []
        for memory_uuid, owner_rank in self.uuid_to_rank_mapping.items():
            if owner_rank == rank:
                uuids.append(memory_uuid)
        return uuids

    def get_memory_regions_for_rank(self, rank: int) -> list[MemoryRegion]:
        """Get all memory regions owned by a specific rank.
        
        Args:
            rank: The rank whose memory regions to retrieve
            
        Returns:
            List of MemoryRegion objects owned by the specified rank
        """
        uuids = self.get_uuids_for_rank(rank)
        regions = []
        for memory_uuid in uuids:
            if memory_uuid in self.memory_regions:
                regions.append(self.memory_regions[memory_uuid])
        return regions

    def get_local_memory_regions(self) -> list[MemoryRegion]:
        """Get all local memory regions.
        
        Returns:
            List of all local memory regions
        """
        return [region for region in self.memory_regions.values() if region.is_local]

    def get_remote_memory_regions(self) -> list[MemoryRegion]:
        """Get all remote memory regions.
        
        Returns:
            List of all remote memory regions
        """
        return [region for region in self.memory_regions.values() if not region.is_local]

    def report_enqueued_requests(self) -> dict[int, list[dict]]:
        """Report enqueued requests for local memory regions to each rank using MPI.
        
        This function collects all enqueued requests from remote memory regions
        and reports them to the ranks that own the target memory regions.
        
        Returns:
            Dictionary mapping rank -> list of request dictionaries containing:
            - uuid: UUID of the target memory region
            - reqtype: RequestType (PUT or GET)
            - timestamp: LogicalTimestamp value
            - offset: Memory offset
            - buffer_size: Size of the buffer in bytes
            - source_rank: Rank that initiated the request
        """
        # Step 1: Collect all enqueued requests from remote memory regions
        remote_requests = []
        for region in self.get_remote_memory_regions():
            for request in region.queue.requests:
                request_info = {
                    'uuid': region.uuid.bytes,  # Convert UUID to bytes for MPI
                    'reqtype': request.reqtype.value,  # Convert enum to int
                    'timestamp': request.timestamp.local_time,
                    'offset': request.offset,
                    'buffer_size': len(request.buffer),
                    'source_rank': self.rank
                }
                remote_requests.append(request_info)
        
        # Step 2: Count requests per rank to prepare for alltoall
        requests_per_rank = {}
        for request in remote_requests:
            target_rank = self.uuid_to_rank_mapping[uuid.UUID(bytes=request['uuid'])]
            if target_rank not in requests_per_rank:
                requests_per_rank[target_rank] = []
            requests_per_rank[target_rank].append(request)
        
        # Step 3: Prepare data for MPI alltoall
        # Each rank sends a list of requests to each other rank
        send_data = []
        for target_rank in range(self.world_size):
            if target_rank in requests_per_rank:
                # Serialize requests for this target rank
                rank_requests = requests_per_rank[target_rank]
                serialized = self._serialize_requests(rank_requests)
            else:
                serialized = b''
            send_data.append(serialized)
        
        # Step 4: Exchange request data using alltoall
        recv_data = self.comm.alltoall(send_data)
        
        # Step 5: Deserialize received requests and organize by source rank
        received_requests = {}
        for source_rank, data in enumerate(recv_data):
            if data:  # Skip empty data
                requests = self._deserialize_requests(data)
                if requests:
                    received_requests[source_rank] = requests
        
        return received_requests
    
    def _serialize_requests(self, requests: list[dict]) -> bytes:
        """Serialize a list of request dictionaries to bytes for MPI communication.
        
        Args:
            requests: List of request dictionaries
            
        Returns:
            Serialized bytes representation
        """
        if not requests:
            return b''
        
        # Format: [num_requests][uuid][reqtype][timestamp][offset][buffer_size][source_rank]...
        data = []
        data.append(len(requests).to_bytes(4, byteorder='big'))
        
        for req in requests:
            data.append(req['uuid'])  # 16 bytes
            data.append(req['reqtype'].to_bytes(4, byteorder='big'))
            data.append(req['timestamp'].to_bytes(8, byteorder='big'))
            data.append(req['offset'].to_bytes(8, byteorder='big'))
            data.append(req['buffer_size'].to_bytes(8, byteorder='big'))
            data.append(req['source_rank'].to_bytes(4, byteorder='big'))
        
        return b''.join(data)
    
    def _deserialize_requests(self, data: bytes) -> list[dict]:
        """Deserialize bytes back to a list of request dictionaries.
        
        Args:
            data: Serialized request data
            
        Returns:
            List of request dictionaries
        """
        if not data:
            return []
        
        requests = []
        offset = 0
        
        # Read number of requests
        num_requests = int.from_bytes(data[offset:offset+4], byteorder='big')
        offset += 4
        
        # Read each request
        for _ in range(num_requests):
            # UUID (16 bytes)
            uuid_bytes = data[offset:offset+16]
            offset += 16
            
            # Request type (4 bytes)
            reqtype = int.from_bytes(data[offset:offset+4], byteorder='big')
            offset += 4
            
            # Timestamp (8 bytes)
            timestamp = int.from_bytes(data[offset:offset+8], byteorder='big')
            offset += 8
            
            # Offset (8 bytes)
            req_offset = int.from_bytes(data[offset:offset+8], byteorder='big')
            offset += 8
            
            # Buffer size (8 bytes)
            buffer_size = int.from_bytes(data[offset:offset+8], byteorder='big')
            offset += 8
            
            # Source rank (4 bytes)
            source_rank = int.from_bytes(data[offset:offset+4], byteorder='big')
            offset += 4
            
            request = {
                'uuid': uuid.UUID(bytes=uuid_bytes),
                'reqtype': RequestType(reqtype),
                'timestamp': LogicalTimestamp(timestamp),
                'offset': req_offset,
                'buffer_size': buffer_size,
                'source_rank': source_rank
            }
            requests.append(request)
        
        return requests

    def process_incoming_requests(self, received_requests: dict[int, list[dict]]) -> dict[uuid.UUID, list[dict]]:
        """Process incoming requests for local memory regions.
        
        This function determines which requests can be completed based on the distance
        matrix and local timestamps, then executes the ready requests.
        
        Args:
            received_requests: Dictionary mapping source_rank -> list of request dictionaries
                             from report_enqueued_requests()
        
        Returns:
            Dictionary mapping memory_region_uuid -> list of completed request info
        """
        completed_requests = {}
        
        # Group requests by target memory region UUID
        requests_by_region = self._group_requests_by_region(received_requests)
        
        # Process each local memory region
        for region_uuid, requests in requests_by_region.items():
            if region_uuid in self.memory_regions:
                memory_region = self.memory_regions[region_uuid]
                if memory_region.is_local:
                    # Determine which requests are ready and execute them
                    ready_requests = self._determine_ready_requests(memory_region, requests)
                    completed = self._execute_ready_requests(memory_region, ready_requests)
                    if completed:
                        completed_requests[region_uuid] = completed
        
        return completed_requests
    
    def _group_requests_by_region(self, received_requests: dict[int, list[dict]]) -> dict[uuid.UUID, list[dict]]:
        """Group incoming requests by target memory region UUID.
        
        Args:
            received_requests: Dictionary mapping source_rank -> list of request dictionaries
        
        Returns:
            Dictionary mapping memory_region_uuid -> list of request dictionaries
        """
        requests_by_region = {}
        
        for source_rank, requests in received_requests.items():
            for request in requests:
                region_uuid = request['uuid']
                if region_uuid not in requests_by_region:
                    requests_by_region[region_uuid] = []
                requests_by_region[region_uuid].append(request)
        
        return requests_by_region
    
    def _determine_ready_requests(self, memory_region: 'MemoryRegion', requests: list[dict]) -> list[dict]:
        """Determine which requests are ready to be executed based on timestamps and distance.
        
        A request is "ready" if its timestamp is between the current time of the memory region
        and the distance to the remote rank.
        
        Args:
            memory_region: The local memory region to check requests for
            requests: List of request dictionaries targeting this memory region
        
        Returns:
            List of request dictionaries that are ready to be executed
        """
        ready_requests = []
        current_time = memory_region.time.local_time
        
        for request in requests:
            source_rank = request['source_rank']
            request_timestamp = request['timestamp'].local_time
            distance = self.get_distance(source_rank, self.rank)
            
            # A request is ready if its timestamp is between current_time and current_time + distance
            if current_time <= request_timestamp <= current_time + distance:
                ready_requests.append(request)
        
        return ready_requests
    
    def _execute_ready_requests(self, memory_region: 'MemoryRegion', ready_requests: list[dict]) -> list[dict]:
        """Execute the ready requests for a memory region.
        
        Args:
            memory_region: The local memory region to execute requests on
            ready_requests: List of request dictionaries that are ready to be executed
        
        Returns:
            List of completed request information
        """
        completed_requests = []
        
        for request in ready_requests:
            try:
                # Execute the request based on its type
                if request['reqtype'] == RequestType.PUT:
                    result = self._execute_put_request(memory_region, request)
                elif request['reqtype'] == RequestType.GET:
                    result = self._execute_get_request(memory_region, request)
                else:
                    # Skip unknown request types
                    continue
                
                # Add completion info
                completion_info = {
                    'source_rank': request['source_rank'],
                    'reqtype': request['reqtype'],
                    'timestamp': request['timestamp'],
                    'offset': request['offset'],
                    'buffer_size': request['buffer_size'],
                    'status': 'completed',
                    'result': result
                }
                completed_requests.append(completion_info)
                
            except Exception as e:
                # Handle execution errors
                completion_info = {
                    'source_rank': request['source_rank'],
                    'reqtype': request['reqtype'],
                    'timestamp': request['timestamp'],
                    'offset': request['offset'],
                    'buffer_size': request['buffer_size'],
                    'status': 'error',
                    'error': str(e)
                }
                completed_requests.append(completion_info)
        
        return completed_requests
    
    def _execute_put_request(self, memory_region: 'MemoryRegion', request: dict) -> dict:
        """Execute a PUT request on a memory region.
        
        Args:
            memory_region: The local memory region to write to
            request: Request dictionary containing PUT operation details
        
        Returns:
            Dictionary with execution result information
        """
        # For now, we'll simulate the PUT operation
        # In a real implementation, this would write to the actual memory
        offset = request['offset']
        buffer_size = request['buffer_size']
        
        # Update the memory region's timestamp to the request timestamp
        memory_region.time.local_time = max(memory_region.time.local_time, request['timestamp'].local_time)
        
        # Simulate writing data (in real implementation, this would write to memory)
        result = {
            'operation': 'PUT',
            'bytes_written': buffer_size,
            'offset': offset,
            'new_timestamp': memory_region.time.local_time
        }
        
        return result
    
    def _execute_get_request(self, memory_region: 'MemoryRegion', request: dict) -> dict:
        """Execute a GET request on a memory region.
        
        Args:
            memory_region: The local memory region to read from
            request: Request dictionary containing GET operation details
        
        Returns:
            Dictionary with execution result information
        """
        # For now, we'll simulate the GET operation
        # In a real implementation, this would read from the actual memory
        offset = request['offset']
        buffer_size = request['buffer_size']
        
        # Update the memory region's timestamp to the request timestamp
        memory_region.time.local_time = max(memory_region.time.local_time, request['timestamp'].local_time)
        
        # Simulate reading data (in real implementation, this would read from memory)
        result = {
            'operation': 'GET',
            'bytes_read': buffer_size,
            'offset': offset,
            'new_timestamp': memory_region.time.local_time
        }
        
        return result

    def clear_processed_requests(self, completed_requests: dict[uuid.UUID, list[dict]]):
        """Clear processed requests from the memory region queues.
        
        This function removes requests that have been processed from the queues
        to prevent them from being processed again in future iterations.
        
        Args:
            completed_requests: Dictionary mapping memory_region_uuid -> list of completed request info
        """
        for region_uuid, requests in completed_requests.items():
            if region_uuid in self.memory_regions:
                memory_region = self.memory_regions[region_uuid]
                if memory_region.is_local:
                    # For now, we'll clear all requests from local memory regions
                    # In a more sophisticated implementation, we would match specific requests
                    memory_region.queue.requests.clear()
                else:
                    # For remote memory regions, we need to match and remove specific requests
                    # that were processed by the owner rank
                    processed_requests = set()
                    for req_info in requests:
                        # Create a key to identify the processed request
                        req_key = (
                            req_info['source_rank'],
                            req_info['reqtype'],
                            req_info['timestamp'].local_time,
                            req_info['offset'],
                            req_info['buffer_size']
                        )
                        processed_requests.add(req_key)
                    
                    # Remove matching requests from the queue
                    memory_region.queue.requests = [
                        req for req in memory_region.queue.requests
                        if (
                            req.rank,
                            req.reqtype,
                            req.timestamp.local_time,
                            req.offset,
                            len(req.buffer)
                        ) not in processed_requests
                    ]

    def advance_local_time(self, delta: int = 1):
        """Advance the local time of all local memory regions.
        
        Args:
            delta: Amount to advance the time by (default: 1)
        """
        for memory_region in self.get_local_memory_regions():
            memory_region.time.local_time += delta

    def process_all_requests(self) -> dict[uuid.UUID, list[dict]]:
        """Main function to process all requests in the system.
        
        This function:
        1. Gathers remote requests to local memory regions
        2. Determines which requests can be completed
        3. Executes the ready requests
        4. Clears processed requests from queues
        5. Returns the results
        
        Returns:
            Dictionary mapping memory_region_uuid -> list of completed request info
        """
        # Step 1: Gather remote requests to local memory regions
        received_requests = self.report_enqueued_requests()
        
        # Step 2: Process incoming requests and execute ready ones
        completed_requests = self.process_incoming_requests(received_requests)
        
        # Step 3: Clear processed requests from queues
        self.clear_processed_requests(completed_requests)
        
        return completed_requests


@final
class LogicalTimestamp:
    """Virtual Time class, for internal use only...although we'll have to figure out how to interface it with the simulation"""

    def __init__(self, now: int = 0):
        self.local_time: int = now

    def __add__(self, delta: int):
        return LogicalTimestamp(self.local_time + delta)
    
    @classmethod
    def from_memory_region(cls, memory_region: 'MemoryRegion') -> 'LogicalTimestamp':
        """Create a timestamp based on the current time of a specific memory region.
        
        Args:
            memory_region: The memory region whose current time to use
            
        Returns:
            A new LogicalTimestamp with the memory region's current time
        """
        return cls(memory_region.time.local_time)


class RequestType(Enum):
    PUT = 1
    GET = 2


@final
class MemoryRegion:
    """Unique memory region that the dstm manages."""

    def __init__(
        self,
        uuid: uuid.UUID,
        timestamp: LogicalTimestamp,
        context: Context,
        memsize: int,
    ):
        self.time = timestamp
        self.uuid = uuid
        self.context = context
        self.memsize = memsize
        self.mems_per_rank = context.mems_per_rank if hasattr(context, 'mems_per_rank') else 1
        # Note: owner_rank will be set after the UUID mapping is shared
        self.owner_rank = None
        self.is_local = None
        self.queue = MemoryOperationsQueue(self)
    
    def _set_owner_info(self):
        """Set the owner rank and is_local flag after UUID mapping is available."""
        if self.uuid in self.context.uuid_to_rank_mapping:
            self.owner_rank = self.context.uuid_to_rank_mapping[self.uuid]
            self.is_local = self.owner_rank == self.context.rank

    def put(self, timestamp: LogicalTimestamp, offset: int, buffer: Buffer):
        return self.queue.add_put_request(timestamp, offset, buffer)

    def get(self, timestamp: LogicalTimestamp, offset: int, buffer: Buffer):
        return self.queue.add_get_request(timestamp, offset, buffer)


@final
class MemoryRequest:
    """Tracking object for a request in flight."""

    def __init__(
        self,
        reqtype: RequestType,
        timestamp: LogicalTimestamp,
        offset: int,
        buffer: Buffer,
        queue: 'MemoryOperationsQueue',
    ):
        self.reqtype = reqtype
        self.offset = offset
        self.buffer = buffer
        self.queue = queue
        self.rank = queue.memory_region.context.rank


@final
class MemoryOperationsQueue:
    """Queue operations to a specific memory region"""

    def __init__(self, memory_region: 'MemoryRegion'):
        self.requests: list[MemoryRequest] = []
        self.memory_region = memory_region

    def add_put_request(self, timestamp: LogicalTimestamp, offset: int, buffer: Buffer):
        req = MemoryRequest(RequestType.PUT, timestamp, offset, buffer, self)
        self.requests.append(req)
        return req

    def add_get_request(self, timestamp: LogicalTimestamp, offset: int, buffer: Buffer):
        req = MemoryRequest(RequestType.GET, timestamp, offset, buffer, self)
        self.requests.append(req)
        return req


if __name__ == "__main__":
    import logging
    import numpy as np
    from mpi4py.util import dtlib
    world_size = MPI.COMM_WORLD.Get_size()
    
    # Default rink topology for easy testing
    distance_matrix = np.zeros((world_size, world_size), dtype=np.int32)
    for i in range(world_size):
        # have to be careful with borders 
        n = (i + 1) % world_size
        p = (i + world_size - 1) % world_size 
        logging.info(f"Rank {i} has neighbors {n} and {p}")
        distance_matrix[i][n] = 1
        distance_matrix[i][p] = 1
   
    # create context, feed the static distance matrix to the dstm
    context = Context(distance_matrix=distance_matrix)
    logging.info(f"Rank {context.rank} of {context.world_size}")

    # create a few local memories, storing each a vector of ints
    datatype = MPI.FLOAT
    np_dtype = dtlib.to_numpy_dtype(datatype)
    itemsize = datatype.Get_size()
    memory_uuids = context.create_memory_regions(context.world_size * itemsize, 2)
    logging.info(f"Rank {context.rank} created memory regions with UUIDs: {memory_uuids}")
    logging.info(f"Rank {context.rank} UUID-to-rank mapping: {context.uuid_to_rank_mapping}")
    local_time = LogicalTimestamp()

    # send a message at time 0 to the memories of the next rank for timestamp 1
    neighbor = (context.rank + 1) % context.world_size
    reqs: list[MemoryRequest] = []
    for uuid in context.get_uuids_for_rank(neighbor):
        memory_region = context.get_memory_region(uuid)
        buf = np.empty(context.world_size, dtype=np_dtype)
        req = memory_region.put(local_time + 1, 0, buf)
        reqs.append(req)

    logging.info(f"Rank {context.rank} has {len(reqs)} enqueued requests")
   
    # this should be a no-op, since we're at time 0 and there's nothing pending
    # for writes here
    completed_requests = context.poll(reqs)
    logging.info(f"Rank {context.rank} has poll result {completed_requests}")

    # ask how far we can make progress in time
    how_far = context.max_time_advance()
    logging.info(f"Rank {context.rank} can advance by {how_far}")

    # inform the context that we are advancing by 1 anyway
    # we could do context.local_time ++, which is actually a problem
    context.step(1)
    logging.info(f"Rank {context.rank} - Advanced local time by 1")

    # set up a list of local reads
    for uuid in context.get_uuids_for_rank(context.rank):
        memory_region = context.get_memory_region(uuid)
        buf = np.empty(context.world_size, dtype=np_dtype)
        req = memory_region.get(context.local_time, 0, buf)
        reqs.append(req)

    # TODO: one at a time or batches? I guess batches will be better
    completed_requests = context.poll(reqs) 
    logging.info(f"Rank {context.rank} has poll result {completed_requests}")

    # I'm not a fan of the current API, since buffers end up not being
    # consumed by requests.
   
