import re
import sys
from typing import Dict, List, Tuple

def parse_network_sim_log(log_data: str) -> List[Dict]:
    timestep_snapshots = []
    pattern = r'Rank\s+(\d+)\s+\[t=(\d+)\]'
    
    lines = log_data.split('\n')
    rank_states = {}
    
    for line in lines:
        matches = re.findall(pattern, line)
        for match in matches:
            rank = int(match[0])
            timestep = int(match[1])
            rank_states[rank] = timestep
        
        if rank_states:
            max_distance = 0
            if len(rank_states) > 1:
                ranks_list = list(rank_states.values())
                max_distance = max(ranks_list) - min(ranks_list)
            
            timestep_snapshots.append({
                'rank_states': dict(rank_states),
                'max_distance': max_distance
            })
    
    return timestep_snapshots

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    with open(log_file, "r") as f:
        log_data = f.read()
    
    snapshots = parse_network_sim_log(log_data)
    
    max_overall = 0
    max_snapshot = None
    
    for snapshot in snapshots:
        if snapshot['max_distance'] > max_overall:
            max_overall = snapshot['max_distance']
            max_snapshot = snapshot
    
    if max_snapshot:
        ranks = max_snapshot['rank_states']
        min_rank = min(ranks, key=ranks.get)
        max_rank = max(ranks, key=ranks.get)
        
        print(f"Maximum time step distance: {max_overall}")
        print(f"Rank {min_rank} at t={ranks[min_rank]} and Rank {max_rank} at t={ranks[max_rank]}")
        print(f"\nAll ranks at that moment:")
        for rank in sorted(ranks.keys()):
            print(f"  Rank {rank}: t={ranks[rank]}")
