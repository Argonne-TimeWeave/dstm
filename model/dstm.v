Require Import String.
Require Import stdpp.gmap.

Definition var := string.
Definition loc := nat.
Definition timestamp := nat.
Definition val := Z.

Inductive instr :=
| Load (t1 : timestamp) (x : var) (l : loc) (t : timestamp)    (* at t1: x <- (l, t)  where  t ≤ t1 *)
(* t must be less than t1, unless l belongs to the issuing node *)
| Store (t1 : timestamp) (l : loc) (t : timestamp) (v : val)   (* at t1: (l, t) <- v  where t1 ≤ t *)
(* likewise, they can only be equal if l belongs to the issuing node *)
.
(* expectation: each node issues instructions with monotonic t1's *)
(* store is non-blocking, load is blocking -- waits for it to become readable because nobody is writing
   it anymore *)
(* for every good program, there should exist a valid schedule *)
(* to signal that there will be no more stores to a location/timestamp:

  Store ... l t1
  Store t2 (t2 > t1) then this node will no longer issue stores that come from t1

  barrier indicates that the node will not send any more instructions
*)
(* when the runtime knows that all nodes will not write (l, t), then (l, t) becomes readable *)

Section semantics.

Definition node_id := nat.
Variable owner : loc -> node_id.

Definition env := gmap var val.
Definition node_pool := gmap node_id (list instr * env).
Inductive mem_status := Writeable | Readable (*| Discardable*).
Definition mem := gmap (loc * timestamp) (val * mem_status).
Definition config := (node_pool * mem)%type.

Inductive step : config -> config -> Prop :=
| step_load np i t1 x l t rest e m v (Hi: np !! i = Some (Load t1 x l t :: rest, e))
  (Hcheck : if owner l =? i then t <= t1 else t < t1) (* this can be checked statically *)
  (Hload: m !! (l, t) = Some (v, Readable)):
    step (np, m) (<[i := (rest, <[x := v]>e)]>np, m)
| step_store np i t1 l t v0 v rest e m (Hi : np !! i = Some (Store t1 l t v :: rest, e))
  (Hcheck : if owner l =? i then t1 <= t else t1 < t)
  (Hload: m !! (l, t) = Some (v0, Writeable)):
    step (np, m) (<[i := (rest, e)]>np, <[(l, t) := (v, Writeable)]>m)
| step_readable np m l t v
  (Hload: m !! (l, t) = Some (v, Writeable)):
  step (np, m) (np, <[(l, t) := (v, Readable)]>m)
(* It's the runtime's responsibility to ensure that this is done at the right time to make the program work. *)
.

End semantics.

(* Examples 

Ring(n):
for some number of timestamps t:
  at t, store (n + 1) (t + 1) (t + 1)
  at t, load n t , make sure the value is t

*)

Section status_semantics.

Variable owner : loc -> node_id.

Definition timestamp_of i :=
  match i with Load t _ _ _ | Store t _ _ _ => t end.

Definition live_nodes (np : node_pool) := dom (filter (λ '(_, (li, _)), li ≠ []) np).
Definition config' := (node_pool * mem * gmap node_id timestamp)%type.

Inductive step' : config' -> config' -> Prop :=
| step'_load np i t1 x l t rest e m v nt (Hi: np !! i = Some (Load t1 x l t :: rest, e))
  (Hcheck : if owner l =? i then t <= t1 else t < t1)
  (Hload: m !! (l, t) = Some (v, Readable)):
    step' (np, m, nt) (<[i := (rest, <[x := v]>e)]>np, m, nt)
| step'_store np i t1 l t v0 v rest e m nt (Hi : np !! i = Some (Store t1 l t v :: rest, e))
  (Hload: m !! (l, t) = Some (v0, Writeable)):
    step' (np, m, nt) (<[i := (rest, e)]>np, <[(l, t) := (v, Writeable)]>m, <[i := t1]>nt)
| step'_readable np m l t v nt
  (Hload: m !! (l, t) = Some (v, Writeable))
  (Hpast: forall i, i ∈ live_nodes np -> exists ti, nt !! i = Some ti /\ t < ti):
  step' (np, m, nt) (np, <[(l, t) := (v, Readable)]>m, nt)
(* As written, this is still optional -- we can prove that it doesn't happen any earlier, but not that it
   happens as soon as it's ready. *)
.

Definition monotonic (li : list instr) := forall a b ia ib, a < b -> li !! a = Some ia -> li !! b = Some ib ->
  timestamp_of ia <= timestamp_of ib.

(* If all nodes' programs are monotonic, then this implementation can do anything that the nondeterministic
   implementation can do. *)
Theorem monotonic_step'_safe np : map_Forall (λ _ '(li, _), monotonic li) np ->
  forall m np' m', rtc (step owner) (np, m) (np', m') ->
  exists nt', rtc step' (np, m, empty) (np', m', nt').
Proof.
Abort.

End status_semantics.

(* further refinement: travel time between nodes, and its relationship to read/write limits *)