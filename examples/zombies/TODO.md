# zombie TODO 

figure out correct value for cell\_capcity

zombies now wander towards the local cell with the highest population
    or "POPULAR" locations
        humans do not avoid zombies currently...

# How to run on multi-node
## This is 8 nodes; 4 ranks per node
mpirun -n 32 --hostfile "$COBALT\_NODEFILE" \
    --prefix "$(dirname "$(dirname "$(which mpirun)")")" \
    --map-by ppr:4:node \
    -x LD\_LIBRARY\_PATH -x PATH \
    ./zombies
