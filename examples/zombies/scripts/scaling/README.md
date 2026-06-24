# scaling results

We get different results with the zombies simulation than we do with the airport simulation. In the airport simulation DSTM was able to scale much better than the MPI version. While in the zombie simulations case we see the opposite (to a lesser extent). 

## Hypothesis
The source of DSTM's *relative performance drop* in the zombie simulation is due to the lack of a *gate* system. In the airport simulation we knew that layovers occurred when a plane needed to travel longer than a Manhattan distance of three. Therefore, each airport only had 24 *gates* as that is how many airports could reach that airport without needing a layover. We were able to use a hash in order to insure that each gate was only accessed by one airport and therefore we would have no cases of planes overwriting eachother. *sidenote, DSTM could add the capacity for collision detection to avoid this problem*. In the zombie simulation we do no such calculating. Therefore, not only does our darray scale poorly, but also our sentiels and heartbeats scale poorly.  

```C
    //main.c....

/* Global darray — one bundle per (dest_rank, src_rank). */
dstm_darray_t *cells;
status = dstm_darray_create_equal(&cells,
                                  sizeof(cell_slot_t),
                                  (size_t)num_ranks * (size_t)num_ranks);

...

    //zombie.c... timestep

/* STEP 1 — sentinel PUT for our self-bundle. We never write real
 * humans into self; this just makes sure GETs don't block. */
{
    size_t idx = bundle_index(rank, rank, num_ranks);
    dstm_darray_put(st->cells, idx, &empty_cell, t, 0);
}
```

## Discussion

I think that this result is interesting and important to note. I also tried to implement the gate system for the zombie simulation as well (since humans only travel far enough to make it to their events on time), but I am struggling with this for now. I want to do some more testing to see how communication counts and timings play into this.

I think that another source of this *relative performance drop* is the payload and message volume. In the zombie sim, payload and message volume are much higher(partly due to the *gate* situation). Humans are much larger structs than within the airport simulation, so the MPI\_Pack and MPI\_Unpack cost may be providing a decent amount of overhead. Since, of course, the MPI version just Isends the raw cell\_slot\_t as MPI\_BYTE. **something else to look into**.
